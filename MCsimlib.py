#!/usr/bin/python


"""
MCsimlib simulates fluorosequencing using a MonteCarlo strategy.
"""


import sys
import time
import math
from math import log, sqrt, ceil
import random
import itertools
import cPickle
#import string
#import logging
#import re
#import gmpy
import operator
#import collections
#import scipy
#import scipy.sparse
import multiprocessing
#import smlr
import datetime
#import randsiggen
#import gc
import csv
import numpy as np
from itertools import combinations_with_replacement, product
from sklearn.mixture import GMM, DPGMM
from scipy.stats import norm, lognorm
from operator import mul
from sklearn.cluster import KMeans
from os.path import basename
#import pflib
from collections import namedtuple, defaultdict
from string import letters, digits


def _dp(d, e, p):
    """
    Bernoulli probability of e delays in a gap length d, given Edman success p.
    """
    #TODO:figure out how to determine which interpreter runs so
    #we can decide to use gmpy or not
    #return int(gmpy.bincoef(d - 1 + e, e)) * p**d * q**e
    #use this for pypy as it doesn't play well with gmpy
    q = 1.0 - p
    return (math.factorial(d - 1 + e) /
            (math.factorial(e) * math.factorial(d - 1)) *
            p**d * q**e)

def load_proteome(filename, silent=True):
    """
    Unpickles proteome and returns it as a dictionary.

    Args:
        filename: String indicating proteome pickle file location.
        silent: Boolean indicating whether progress should be printed to
            standard output.

    Returns:
        Dictionary containing proteome.
        {'PROTEIN NAME 1': 'AMINO ACID SEQUENCE 1',
         'PROTEIN NAME 2': 'AMINO ACID SEQUENCE 2',
         ...}
    """
    load_proteome_time = time.clock()
    if not silent:
        print("")
        print("unpickling proteome from " + str(filename))
    #the expected proteome dictionary format in the pickled file is
    #{'PROTEIN NAME': 'AMINO ACID SEQUENCE'}
    try:
        proteome = cPickle.load(open(filename))
    except:
        print("Proteome unpickling error.")
        raise
    if not silent:
        #using time.clock() - starting_time is prone to underflow after ~30 min
        print("proteome loaded in " + str(time.clock() - load_proteome_time) +
              " sec")
        print("...")
    return proteome

def homogenize(peptides, substitute_acid, target_acids):
    """
    Replace each instance of every acid in target_acids with a substitute acid.
    Its original intended use is to simplify generating multichroic signals
    where one color labels more than one amino acid.

    Args:
        peptides: Dictionary of peptides to be homogenized. It is in the same
            format as the dictionary returned by load_proteome.
            {'PROTEIN 1': 'AMINO ACID SEQUENCE 1',
             'PROTEIN 2': 'AMINO ACID SEQUENCE 2',
             ...}
        substitute_acid: One-letter string indicating which amino acid will
            substitute for the target acids.
        target_acids: Tuple or list of one-letter strings indicating which
            amino acids will be substituted with substitute_acid. Example:
            Aspartic acid 'D' will be labeled with the same color as glutamic
            acid 'E'. First call this function with arguments
            substitute_acid='E', target_acids=['D'].

    Returns:
        Dictionary in identical to the peptides dictionary given, except with
        all target_acids replaced with the substitute_acid. This dictionary is
        generated fresh, and does not side-affect the original argument.
    """
    return_peptides = {}
    for protein in peptides:
        sequence = peptides[protein]
        for acid in target_acids:
            homogenized_sequence = sequence.replace(acid, substitute_acid)
        return_peptides.setdefault(protein, homogenized_sequence)
    return return_peptides

def cleave(peptides, cleave_acid, silent=True):
    """
    Cleave each peptide passed to the function at the peptide bond following a
    given amino acid.

    Args:
        peptides: Dictionary of peptides to be cleaved. It is in the same
            format as the dictionary returned by load_proteome.
            {'PROTEIN 1': 'AMINO ACID SEQUENCE 1',
             'PROTEIN 2': 'AMINO ACID SEQUENCE 2',
             ...}
        cleave_acid: One-letter string indicating after which acid to cleave
            the bond, e.g. passing a 'K' cleaves after every lyseine.
        silent: Boolean indicating whether progress should be printed to
            standard output.

    Returns:
        Returns a dictionary mapping the given proteins to the peptides
        resulting from cleaving.
            {'PROTEIN 1': ('CLEAVED SEQUENCE 1', 'CLEAVED SEQUENCE 2', ...),
             'PROTEIN 2': ('CLEAVED SEQUENCE 1', 'CLEAVED SEQUENCE 2', ...),
             ...}
        Empty sequences, i.e. '', are not added. If a protein has no non-empty
        subsequences, its entry is not added. If cleaving a peptide results in
        more than one identical subsequence, they are all added anyways. The
        returned dictionary is generated de novo; the original dictionary
        argument is not affected.
    """
    #progress tracking
    cleave_time = time.clock()
    cleave_progress = 0
    #the new dictionary that will be returned
    return_peptides = {}
    for protein in peptides:
        #disregard empty sequences
        if not peptides[protein]:
            continue
        subsequences = peptides[protein].split(cleave_acid)
        #split(cleave_acid) omits the cleave_acid itself from every resulting
        #substring and adds an empty substring to the end if cleave_acid is at
        #the end of a given sequence, e.g.
        #'ABCABCABCCCC'.split('C') returns ['AB', 'AB', 'AB', '', '', '', '']
        for index, string in enumerate(subsequences[:-1]):
            #add omitted cleave_acid to all but the last gap; if the last acid
            #in the sequence is cleave_acid itself, it leaves an empty
            #subsequence last which needs to be removed anyways; if the last
            #acid is not cleave_acid, then cleave_acid does not need to be
            #added to it
            subsequences[index] += cleave_acid
        if subsequences[-1] == '':
            #if last subsequence resulting from split is empty, this means that
            #the last item in the original sequence was cleave_acid, which is
            #readded by the loop directly above to the penultimate member
            subsequences.pop()
        #eliminate all empty subsequences
        subsequences = [subsequence for subsequence in subsequences
                        if subsequence]
        #if subsequences is empty list, do not add
        if subsequences:
            return_peptides.setdefault(protein, tuple(subsequences))
        if not silent:
            cleave_progress += 1
            sys.stdout.write("%d of %d peptides cleaved\r" %
                             (cleave_progress, len(peptides)))
    if not silent:
        #using time.clock() - starting_time is prone to underflow after ~30 min
        print("")
        print("proteome cleaved in " + str(time.clock() - cleave_time) + "sec")
        print("...")
    return return_peptides

def attach(peptides, attach_acid, silent=True):
    """
    Attach peptides to substrate via an acid.

    Args:
        peptides: Dictionary of peptides to attach. It is in the same format as
            the return value of cleave()
            {'PROTEIN 1': ('CLEAVED SEQUENCE 1', 'CLEAVED SEQUENCE 2'),
             'PROTEIN 2': ('CLEAVED SEQUENCE 1', 'CLEAVED SEQUENCE 2'), ...}
        attach_acid: One-letter string indicating which acid is used to attach
            the peptides, e.g. 'C' will attach all peptides using a cysteine.
            If the string is 'cterm', then all peptides are attached using
            their carboxyl terminus.
        silent: Boolean indicating whether progress should be printed to
            standard output.

    Returns:
        Returns a dictionary mapping proteins to all attached peptides
        associated with them. Those peptides that do not contain the attaching
        acid are omitted. Proteins with no peptides that can attach are
        omitted. Each attached peptide is partitioned in two: the peptide head,
        which represents the portion of the peptide preceding the first
        attaching acid and hence accessible to Edman chemistry, and peptide
        tail, which represents all amino acids after the first attachment,
        where they are inaccessible to Edman chemistry. Returned dictionary
        format is as follows:
            {'PROTEIN 1': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             'PROTEIN 2': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             ...}
        If a protein has more than one identical attachment pair, they are all
        included. This function does not affect the original peptide argument.
    """
    #progress tracking
    attach_time = time.clock()
    attach_progress = 0
    if not silent:
        print("attaching peptides using " + str(attach_acid))
    #the dictionary that will be returned
    return_peptides = {}
    #special case for cterm
    if attach_acid == 'cterm':
        for protein in peptides:
            for index, sequence in enumerate(peptides[protein]):
                return_peptides.setdefault(protein, []).append((sequence,''))
            return_peptides[protein] = tuple(return_peptides[protein])
        return return_peptides
    for protein in peptides:
        for index, sequence in enumerate(peptides[protein]):
            #if sequence does not contain attaching acid, it is omitted
            if attach_acid in sequence:
                attach_point = sequence.find(attach_acid)
                return_peptides.setdefault(protein, [])
                return_peptides[protein].append((sequence[:attach_point],
                                                 sequence[attach_point:]))
        if protein in return_peptides:
            #if protein has successful attachments, tuplefy result
            return_peptides[protein] = tuple(return_peptides[protein])
        if not silent:
            attach_progress += 1
            sys.stdout.write("peptides for %d of %d proteins attached\r" %
                             (attach_progress, len(peptides)))
    if not silent:
        #using time.clock() - starting_time is prone to underflow after ~30 min
        print("")
        print("peptides attached in " + str(time.clock() - attach_time) +
              " sec")
        print("...")
    return return_peptides

def homogenize_attached(peptides, substitute_acid, target_acids):
    """
    Same as homogenize, but operates on peptides in the format yielded by
    attached.
    """
    return_peptides = {}
    for protein, sequences in peptides.iteritems():
        for head, tail in sequences:
            for acid in target_acids:
                head = head.replace(acid, substitute_acid)
                tail = tail.replace(acid, substitute_acid)
            return_peptides.setdefault(protein, []).append((head, tail))
    for protein, sequences in return_peptides.iteritems():
        return_peptides[protein] = tuple(sequences)
    return return_peptides

def discard(peptides, label_acids, (tot_min, tot_max), silent=True):
    """
    Discard all sequences that do not have the indicated number of labelling
    amino acids in their head segment. This function was originally intended to
    group peptides by number of labels, and to eliminate peptides with an
    excessive number of labels whose combinatorics would overwhelm my laptop.

    Args:
        peptides: Dictionary of peptides to be culled. It is in the same
            format as the dictionary returned by attach.
            {'PROTEIN 1': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             'PROTEIN 2': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             ...}
        label_acids: Tuple or array of label acids, e.g. ['K', 'E']
        (tot_min, tot_max): Tuple indicating total maximum and minimum number
            of all labels of any kind on the head segment of each peptide.
        silent: Boolean indicating whether progress should be printed to
            standard output.

    Returns:
        Dictionary identical in format to that given as peptides, except with
        those peptides not satisfying the criteria omitted. Those proteins
        which do not have any peptides satisfying the criteria are omitted.
        Duplicate sequences for each protein are processed without discarding.
        The argument dictionary is unaffected.
    """
    raise DeprecationWarning
    #TODO: We may need to test for tail acids in the future as well.
    #TODO: Individual label acid criteria, e.g. {'K': (3,4), 'E': (5,6)}
    #progress tracking
    discard_time = time.clock()
    discard_progress = 0
    if not silent:
        print("discarding peptides using " + str(label_acids))
    #this will be the returned dictionary
    return_peptides = {}
    for protein in peptides:
        #for each protein, filter all satisfying peptides into culled_sequences
        #add a peptide only if it is within (tot_min, tot_max) inclusive number
        #of labels on head segment
        culled_sequences = [sequence for sequence in peptides[protein]
                            if (tot_min
                                <= sum(sequence[0].count(acid)
                                       for acid in label_acids) <=
                                tot_max)]
        #if no sequences pass muster, do not add the protein at all
        if culled_sequences:
            return_peptides.setdefault(protein, tuple(culled_sequences))
        if not silent:
            discard_progress += 1
            sys.stdout.write("peptides discarded for %d of %d proteins\r" %
                             (discard_progress, len(peptides)))
    if not silent:
        print("")
        print("peptides discarded in " + str(time.clock() - discard_time) +
              " sec")
        print("...")
    #using time.clock() - starting_time is prone to underflow after ~30 min
    return return_peptides

def truncate_heads(peptides, max_edmans):
    """
    Like discard(), this function was made to reduce processing time for
    edman_failure_gaps. Supposing we want to simulate a maximum of only
    max_edmans Edman cycles, we can place any amino acids after this number in
    the tails. If these amino acids end up labeled, then all we worry about is
    them not bleaching, as opposed to permuting through all the delays they can
    cause.

    Args:
        peptides: Dictionary of peptides to have their heads truncated. It is
            in the same format as the dictionary returned by attach.
            {'PROTEIN 1': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             'PROTEIN 2': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             ...}
        max_edmans: Integer indicating the maximum number of amino acids left
            in the head sections of the dictionary.

    Returns:
        Dictionary of peptides identical to the argument peptides, however with
            all head amino acids after max_edmans transferred to the beginning
            of their corresponding tails.
    """
    raise DeprecationWarning
    return_peptides = {}
    for protein, sequences in peptides.iteritems():
        head_truncated_sequences = []
        for sequence in sequences:
            if len(sequence[0]) > max_edmans:
                head_truncated_sequences.append(
                                      (sequence[0][:max_edmans],
                                       sequence[0][max_edmans:] + sequence[1]))
            else:
                head_truncated_sequences.append(sequence)
        return_peptides.setdefault(protein, tuple(head_truncated_sequences))
    return return_peptides

def edman_failure_gaps(peptides, label_acids, p, probability_threshold=0.1,
                       result_queue=None, child_number=0, silent=True):
    """
    Generates signals for all peptides with the likeliest combinations of Edman
    failure. Photobleaching is not modeled. Edman reactions are assumed to
    proceed for every peptide until an anchoring acid.

    Args:
        peptides: Dictionary of peptides used as signal sources. It is in the
            same format as the dictionary returned by attach or discard.
            {'PROTEIN 1': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             'PROTEIN 2': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             ...}
        label_acids: Tuple or list of single-letter amino acids that are to be
            labeled. Each amino acid is considered to be labeled with a color
            distinct from all other labeled acids. If more than one amino acid
            will be labeled with the same color, homogenize the sequences
            first. For example, ['K', 'E'] generates signal combinations with
            lysene as one color and glutamic acid as another.
        p: Probability of an Edman reaction succeeding. All Edman reactions are
            modelled as independent Bernoulli variables. p must be a number in
            [0, 1] inclusive.
        probability_threshold: Probability below which a signal is omitted. The
            probability of each signal is the product of the probability of
            each of its gaps based on Bernoulli Edman failure.
        result_queue: Used by edman_failure_gaps_MP. If None, returns via
            return. If a multiprocessing.Queue is given, puts the results into
            it.
        child_number: Only affects progress outputs by assigning this instance
            of edman_failure_gaps a child number. This is useful if using
            edman_failure_gaps_MP.
        silent: Boolean indicating whether progress should be printed to
            standard output.

    Returns:
        Nested dictionary of the likeliest gaps caused by Edman failure. Each
        gap combination is a tuple of the cumulative distance to a label and
        that label's amino acid, which is assumed to have a one-to-one
        correspondance with a label color. For example, ((4, 'K'), ('7', 'E'))
        represents a combination with a lyseine revealed after the fourth Edman
        cycle followed by a glutamic acid revealed after the seventh Edman
        cycle. The number of cycles does not necessarily correspond with the
        actual location of the acid of course -- that problem is the underlying
        reason for this function. With the gap combinations being used as keys
        in the dictionary, they are each mapped to a dictionary of all the
        proteins that can cause that particular gap combination. With the
        proteins being keys, they point to a tuple of all (tail, probability)
        pairs for that protein that can yield the head gap combination in
        question. The tail is the unmodified tail peptide sequence still
        represented as a string, and probability is the chance that particular
        peptide for that protein can generate the head gap combination.
        {((gap1, label1), (gap2, label2), ...):
                                   {protein1: ((tail1, p1), (tail2, p2)), ...}}
        The item signal_to_protein[signal][protein][i][1] is the probability
        that protein has a cleaved and attached peptide subsequence that
        generates signal from its head and has a tail
        signal_to_protein[signal][protein][i][0]. Peptides that cannot generate
        a signal with a probability above probability_threshold are omitted,
        and proteins that do not have any peptide subsequences qualifying are
        omitted as well. The tails in (tail, probability) tuples are not unique
        for every protein; a given protein may have more than one tuple with
        identical tails.
    """
    raise DeprecationWarning
    #progress tracking
    gap_time = time.clock()
    gap_progress = 0
    #ensure edman probability is a float
    p = float(p)
    #returned dictionary
    signal_to_protein = {}
    for protein in peptides:
        if not silent:
            gap_progress += 1
            if (gap_progress - 1) % 1000 == 0 and gap_progress > 1:
                print("child number " + str(child_number) + ": " +
                      str(gap_progress - 1) + " proteins computed")
        for peptide in peptides[protein]:
            #head of the peptide from which gap combinations will be generated
            head = peptide[0]
            #ideal_gaps is the sequence of gaps under ideal (p=1) conditions
            #stored as a list of pairs (gap, label). each gap is a cumulative
            #gap from the beginning of the peptide.
            #iterate through each item in head, adding a gap location to
            #ideal_gaps whenever a label_acid is found; index + 1 is used as
            #the cumulative distance as python indexing starts at 0.
            #tuplification is profelactic against bugs
            ideal_gaps = tuple([(index + 1, acid)
                                for index, acid in enumerate(head)
                                if acid in label_acids])
            #bug found: if the peptide head has no label_acids in its head, g_e
            #results as []. this causes gap_cartesian_product below becomes
            #[()], which allows for product in gap_cartesian_product to iterate
            #on an empty product, causing an error. fix: test for empty
            #ideal_gaps
            if not ideal_gaps:
                #avoid empty cases; see bug comment above
                continue
            #g_e is array of all possible delays for each gap that are above
            #probability_threshold
            g_e = [[] for x in ideal_gaps]
            #bernoulli probability of total gap length being d + e, where d is
            #ideal gap length and e is delay
            for index, g in enumerate(ideal_gaps):
                #convert cumulative gap to difference gap for probability
                #formula
                d = (ideal_gaps[index][0]  - ideal_gaps[index - 1][0]
                     if index > 0 else ideal_gaps[index][0])
                #probability that a particular d + e is possible.
                #once this falls below probability_threshold, further delays
                #are even less likely. no need to test if e=0 will result in
                #lower than probability_threshold, because 0.97**100 ~ 0.0476 >
                #probability_threshold for even long gaps. each tuple is
                #(local gap, label acid, probability of d + e local gap)
                g_e[index] = [(d + e, g[1], _dp(d, e, p)) for e in
                              itertools.takewhile(lambda x: _dp(d, x, p) >=
                                                  probability_threshold,
                                                  itertools.count())]
            #all possible combinations of gaps for this peptide
            gap_cartesian_product = itertools.product(*g_e)
            for product in gap_cartesian_product:
                #signal produced by this particular combination,
                #(cumulative gap, label acid)
                signal = tuple([(sum(loc[0] for loc in product[:index + 1]),
                                 gap[1])
                                for index, gap in enumerate(product)])
                #probability of this signal is probability of all its gaps
                #occuring
                probability = reduce(operator.mul, [gap[2] for gap in product])
                #add to dictionary if this peptide yields a signal with a gap
                #combination probability above probability_threshold; more than
                #one tuple with identical tails may be added
                if probability >= probability_threshold and signal:
                    signal_to_protein.setdefault(signal,
                                                 {}).setdefault(protein, [])
                    signal_to_protein[signal][protein].append((peptide[1],
                                                               probability))
    #tuplefy innermost array and return; avoiding list comprehension due to
    #possible memory constraints
    for signal in signal_to_protein:
        for protein in signal_to_protein[signal]:
            signal_to_protein[signal][protein] = \
                                      tuple(signal_to_protein[signal][protein])
    if not silent:
        #using time.clock() - starting_time is prone to underflow after ~30 min
        print("child number " + str(child_number) + ": gaps for " +
              str(gap_progress) + " proteins computed in " +
              str(time.clock() - gap_time))
    if result_queue is None:
        return signal_to_protein
    else:
        result_queue.put(signal_to_protein)

def _split_peptides_for_mp(peptides, child_count):
    """
    Evenly partition a proteins in a dictionary of peptides into child_count
    lists. The numbers of proteins in every list will be equal unless the total
    number of proteins is not evenly divisible by child_count, in which case
    the remainder proteins will be allocated one each to some of the lists. If
    child_count is greater than the number of proteins to partition, then
    the proteins will be partitioned into a lower number of lists than
    child_count of one protein each.
 
    Args:
        peptides: Dictionary of peptides used as signal sources. It is in the
            same format as the dictionary returned by attach or discard.
            {'PROTEIN 1': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             'PROTEIN 2': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             ...}
        child_count: Integer indicating into how many lists to partition the
            proteins.

    Returns:
        A list of lists of proteins.
    """
    #ppcb = proteins per child base
    ppcb = len(peptides) / child_count
    #ppcr = proteins per child remainder
    ppcr = len(peptides) % child_count
    proteins = peptides.keys()
    child_protein_list = [[protein
                           for protein in proteins[(ppcb + 1) * x:
                                                   (ppcb + 1) * (x + 1)]]
                          for x in xrange(ppcr)]
    child_protein_list = (child_protein_list +
                              [[protein for protein in
                                  proteins[ppcr * (ppcb + 1) + x * ppcb:
                                           ppcr * (ppcb + 1) + (x + 1) * ppcb]]
                               for x in xrange(child_count - ppcr)])
    return child_protein_list

def edman_failure_gaps_MP(peptides, label_acids, p, probability_threshold=0.1,
                          child_count=None, silent=True):
    """
    Wrapper for running edman_failure_gaps on multiple processors. Functions
    identically to edman_failure_gaps; for usage, see documentation for the
    original single-processor variant.

    Args:
        child_count: Integer that forces spawning of
            min(child_count, len(peptides)) children.
    """
    raise DeprecationWarning
    child_count = (multiprocessing.cpu_count() if child_count is None
                   else min(child_count, len(peptides)))
    child_protein_list = _split_peptides_for_mp(peptides, child_count)
    if not silent:
        print("number of proteins: " + str(len(peptides)) + "; cpu count: " +
              str(child_count) + "; spawning children with " +
              str([len(x) for x in child_protein_list]) +
              " proteins each")
    result_queue = multiprocessing.Queue()
    signal_to_protein = {}
    child_processes = []
    for c, child_proteins in enumerate(child_protein_list):
        #TODO:dictionary comprehensions unavailable in python 2.6; reverting to
        #setdefault
        #child_peptides = {protein: peptides[protein]
        #                  for protein in child_proteins}
        child_peptides = {}
        for protein in child_proteins:
            child_peptides.setdefault(protein, peptides[protein])
        #TODO:end reversion
        child_process = multiprocessing.Process(target=edman_failure_gaps,
                                                args=(child_peptides,
                                                      label_acids, p,
                                                      probability_threshold,
                                                      result_queue, c, silent))
        child_process.start()
        child_processes.append(child_process)
    results = []
    for child in child_processes:
        results.append(result_queue.get())
    for child in child_processes:
        child.join()
    for result in results:
        for signal, proteins in result.iteritems():
            signal_to_protein.setdefault(signal, {}).update(proteins)
    return signal_to_protein

def _exposures(position, windows):
    """
    For any position along a signal peptide, find the number of exposures of
    each color from a given pattern of laser excitations.

    Args:
        position: Position of the amino acid for which the number of exposures
            needs to be determined. As always, position indexing on peptides in
            this simulation starts at 1 to reflect the fact that the first
            observable difference due to Edman cleaving of a labeled acid
            inherently needs at least one reaction. For example, if position =
            3, and exposures are applied for every cycle, then the total number
            of exposures for a color will be 3: the initial exposure before any
            Edman reactions, the exposure after the first Edman, and the
            exposure after the second Edman.
        windows: Laser excitations patterns for each color stored as a
            dictionary. Keys are single-letter strings representing the labeled
            acids, and values are lists or tuples representing the sequence of
            windows during which peptide luminosities are observed. Each member
            of the sequence refers to the number of the Edman cycle for which a
            difference in luminosity is searched for. For example, (3, 4, 7)
            would represent the search for luminosity drops due to the third,
            fourth, and seventh Edman cycle. This means there will be an
            exposure between the second and third Edman, the third and fourth
            Edman, between the fourth and fifth Edman, followed by exposures
            directly before and after the seventh Edman. If position 1 is
            indicated, there is an initial exposure before any Edmans. Thus,
            the dictionary format is e.g. {'E': (3, 4, 7), 'K': (7, 8, 9)}.
            Each amino acid is assumed to be labeled with a distinct color. If
            more than one amino acid is labeled with the same color,
            homogenize() needs to be applied.

    Returns:
        Number of exposures of each color represented as a dictionary
        {acid: number, ...}, e.g. {'E': 5, 'K': 4} would mean that glutamic
        acid's and lyseiene's excitation lasers would be fired five and four
        times, respectively, before the amino acid located at position would be
        cleaved. Only those acids included in the windows argument will have
        results included. This may include the case of no exposures if there
        are no appropriate laser firings before the position is reached.
    """
    #the returned dictionary
    exposure_dictionary = {}
    for acid in windows:
        #list of all exposures for this acid. each exposure is stored as an
        #integer corresponding to the number of the Edman cycle after which it
        #takes place. If there is an initial exposure, it is assigned a 0. It
        #is filled by adding an exposure before and after every windowing
        #position; overlaps are taken care of by set uniqueness property.
        exposures = list(set([x for x in windows[acid]] +
                             [x - 1 for x in windows[acid]]))
        #count number of exposures prior to position
        exposure_dictionary.setdefault(acid, sum(x < position
                                                 for x in exposures))
    return exposure_dictionary

def window_filter(signals, windows):
    """
    Filters signals through windows.

    Args:
        signals: List or tuple of signals.
        windows: Dictionary of windows for each color.

    Returns:
        Tuple of windowed signals. The original list is not modified. Each
        entry in the signals argument has a corresponding element in the
        returned tuple; signal duplication is perpetuated. If a signal yields
        an empty signal due to windowing, it is added as an empty signal to the
        returned value. A luminosity drop needs exposures both directly before
        and directly after to be observable; if it does not satisfy this
        criterion, it is omitted.
    """
    #the returned value
    windowed_signals = []
    for signal in signals:
        #the new filtered signal
        filtered_gaps = []
        for acid in windows:
            exposures = set([x for x in windows[acid]] +
                            [x - 1 for x in windows[acid]])
            filtered_gaps.extend([gap for gap in signal
                                  if (gap[1] == acid and
                                      gap[0] in exposures and
                                      gap[0] - 1 in exposures)])
        #remove identical entries
        filtered_gaps = list(set(filtered_gaps))
        #sort by position and append to returned array; sorted works on empty
        #lists
        filtered_gaps = sorted(filtered_gaps,
                               key=lambda filtered_gaps: filtered_gaps[0])
        windowed_signals.append(tuple(filtered_gaps))
    return tuple(windowed_signals)

def perfect(signal_to_protein, b, windows, probability_threshold=0.01):
    """
    Compute a signal_to_protein dictionary reflecting windowing and
    photobleaching.

    Args:
        signal_to_protein: Dictionary as returned by edman_failure_gaps.
            {((gap1, label1), (gap2, label2), ...):
                                   {protein1: ((tail1, p1), (tail2, p2)), ...}}
            The item signal_to_protein[signal][protein][i][1] is the
            probability that protein has a cleaved and attached peptide
            subsequence that generates signal from its head and has a tail
            signal_to_protein[signal][protein][i][0].
        b: Photobleaching survival constant. Photobleaching is modeled as an
            exponential survival function s(k) = e^-bk, modeling the
            probability of a fluor surviving k laser exposures.
        windows: Laser excitations patterns for each color stored as a
            dictionary. Keys are single-letter strings representing the labeled
            acids, and values are lists or tuples representing the sequence of
            windows during which peptide luminosities are observed. Each member
            of the sequence refers to the number of the Edman cycle for which a
            difference in luminosity is searched for. For example, (3, 4, 7)
            would represent the search for luminosity drops due to the third,
            fourth, and seventh Edman cycle. This means there will be an
            exposure between the second and third Edman, the third and fourth
            Edman, between the fourth and fifth Edman, followed by exposures
            directly before and after the seventh Edman. If position 1 is
            indicated, there is an initial exposure before any Edmans. Thus,
            the dictionary format is e.g. {'E': (3, 4, 7), 'K': (7, 8, 9)}.
            Each amino acid is assumed to be labeled with a distinct color. If
            more than one amino acid is labeled with the same color,
            homogenize() needs to be applied. Having windows corresponding to
            large numbers of Edman reactions means reactions will keep being
            applied that long; the tails need to survive all of these.
        probability_threshold: Probability below which dictionary entries will
            be omitted. If a protein has no qualifying (tail, probability)
            tuples, then it is omitted from the dictionary. Likewise, if a
            windowed signal has no qualifying proteins, it is omitted from the
            dictionary. This implies an empty dictionary may be returned.

    Returns:
        Dictionary in the same format as signal_to_protein, however the signal
        keys are windowed, i.e. only those luminosity drops that would be
        observable due to the given pattern of laser excitation are included in
        a signal. Those signals that are not observable within the given
        windows are omitted from the dictionary. For a luminosity drop in a
        color to be observed during an Edman reaction, the appropriate laser
        must be fired directly before and directly after the reaction.
        (question: can a scheme be built around drops occuring during larger
        gaps?)
    """
    raise DeprecationWarning
    #the dictionary to be returned
    windowed_to_protein = {}
    #ensure photobleaching constant is float in calculations
    b = float(b)
    #exposure_dictionary compiles all of the laser exposures based on the given
    #windowing sequence. Stored as {'acid': (exposures)}, where acid is single-
    #letter string representing the acid, and exposures is a tuple of integers
    #each correspding to the number of the Edman cycle after which each
    #exposure takes place. An initial exposure before any Edman cycles is a 0,
    #an exposure after the first Edman cycle is 1, and so on. implementation is
    #same as in _exposures(), however exposure_dictionary here corresponds to a
    #data structure storing each exposure for every acid rather than the total
    #number of exposures for all acids given a position
    #TODO:dictionary comprehensions unavailable in python 2.6; reverting to
    #setdefault
    #exposure_dictionary = \
    #                   {acid: sorted(list(set([x for x in windows[acid]] +
    #                                         [x - 1 for x in windows[acid]])))
    #                    for acid in windows}
    exposure_dictionary = {}
    for acid in windows:
        exposure_dictionary.setdefault(acid,
                             sorted(list(set([x for x in windows[acid]] +
                                             [x - 1 for x in windows[acid]]))))
    #TODO:end reversion
    for signal in signal_to_protein:
        #filtering for only those luminosity drops that occur within given
        #windows generates the windowed signal
        windowed_signal = tuple([drop for drop in signal
                                 if drop[0] in windows[drop[1]]])
        #TODO:tail only peptides may bleach; create function simulating
        #photobleaching of tail only polypeptides. first step is filtering for
        #all proteins and peptides with only tail labels
        #TODO:peptides whose windowed signal is empty, but can have
        #photobleaching
        #empty windowed_signals are omitted
        if not windowed_signal:
            continue
        #compute updated probabilities of surviving photobleaching for every
        #(tail, probability) tuple.
        #this probability is e**(-b * (sum of exposures for every fluor)), so
        #it is best to first compute the sum first and then exponentiate.
        #first, compute probability of all head fluors surviving; this must
        #include even those not included in windows as they may bleach as well.
        #bug found:this was originally written with x < drop[0] instead of
        #x < drop[0] - 1, which meant this function was expecting the fluor to
        #survive the photobleaching due to laser exposure directly prior to
        #cleaving, when in reality it is not necessary to survive this as
        #either way the luminosity will drop. I may need to write a regression
        #test for this
        head_survival_sum = sum(sum(x < drop[0] - 1
                                    for x in exposure_dictionary[drop[1]])
                                for drop in signal)
        #now for every protein, iterate through its (tail, probability tuples)
        #and generate updated versions for each to put into windowed_to_protein
        for protein in signal_to_protein[signal]:
            for tail in signal_to_protein[signal][protein]:
                #survival sum for each peptide needs to add number of fluors of
                #each color multiplied by number of exposures for that color
                #bug found: write regression unit test for bug found.
                #origianlly was multiplied by len(expsosure_dictionary[acid])
                #instead of len(exposure_dictionary[acid]) - 1, which meant
                #that tail acids needed to survive the laser exposure after
                #the last Edman, which is uneccessary. This is similar to the
                #bug above for the head acids.
                survival_sum = (head_survival_sum +
                  sum(tail[0].count(acid) * (len(exposure_dictionary[acid]) - 1)
                      for acid in exposure_dictionary))
                #total probability of this peptide having this particular
                #windowing sequence and surviving
                total = tail[1] * math.e**(-b * survival_sum)
                #only add if it is above probability_threshold
                if total >= probability_threshold:
                    windowed_to_protein.setdefault(windowed_signal, {}).\
                                                      setdefault(protein, []).\
                                                      append((tail[0], total))
    #tuplefy all arrays for each protein
    for signal in windowed_to_protein:
        for protein in windowed_to_protein[signal]:
            windowed_to_protein[signal][protein] = \
                                    tuple(windowed_to_protein[signal][protein])
    return windowed_to_protein

def random_signal(peptide, p=1.0, b=0.0, u=0.0, windows={}):
    """
    Generate random sequence of luminosity drops from a peptide.

    Args:
        peptide: The subject peptide represented as a tuple ('head string',
            'tail string'), where head string is the sequence of amino acids
            before the first amino acid attached to the substrate, and tail
            string is the sequence of amino acids after and including this
            attaching acid. attach() generates such pair tuples.
        p: Probability of an Edman reaction succeeding. All Edman reactions are
            modelled as independent Bernoulli variables. p must be a number in
            [0, 1] inclusive.
        b: Photobleaching survival constant. Photobleaching is modeled as an
            exponential survival function s(k) = e^-bk, modeling the
            probability of a fluor surviving k laser exposures.
        u: Probability of a fluor being photobleached or unattached or
            otherwise nonfunctional and hence unobservable to begin with.
            Probabilities of each fluor being unobservable are independent.
        windows: Laser excitations patterns for each color stored as a
            dictionary. Keys are single-letter strings representing the labeled
            acids, and values are lists or tuples representing the sequence of
            windows during which peptide luminosities are observed. Each member
            of the sequence refers to the number of the Edman cycle for which a
            difference in luminosity is searched for. For example, (3, 4, 7)
            would represent the search for luminosity drops due to the third,
            fourth, and seventh Edman cycle. This means there will be an
            exposure between the second and third Edman, the third and fourth
            Edman, between the fourth and fifth Edman, followed by exposures
            directly before and after the seventh Edman. If position 1 is
            indicated, there is an initial exposure before any Edmans. Thus,
            the dictionary format is e.g. {'E': (3, 4, 7), 'K': (7, 8, 9)}.
            Each amino acid is assumed to be labeled with a distinct color. If
            more than one amino acid is labeled with the same color,
            homogenize() needs to be applied. Having windows corresponding to
            large numbers of Edman reactions means reactions will keep being
            applied that long; the tails need to survive all of these.

    Returns:
        Returns a sequence of luminosity drops as a tuple. Note: due to
        photobleaching it is possible to have more than one color experience a
        lumnosity drop at the same time, e.g. ((3, 'K'), (3, 'E')) is possible.
    """
    #ensure constants used in calculations are floats
    p, b, u = float(p), float(b), float(u)
    #remove unobservable fluors from signal
    #lowercase 'x' is used as a placeholder for an amino acid with a dead fluor
    for acid in windows:
        #split_head and split_tail: split head and tail using acid
        s_h = peptide[0].split(acid)
        s_t = peptide[1].split(acid)
        #generate random substitution arrays with instances of the acid
        #replaced according to probability u
        h_r = [acid if random.random() > u else 'x'
               for x in range(len(s_h) - 1)]
        t_r = [acid if random.random() > u else 'x'
               for x in range(len(s_t) - 1)]
        #pad with empty strings to ensure split head/tail and replacement
        #head/tail are of same length so that joining them via zip works below
        if len(s_h) > len(h_r):
            h_r += ('',)
        elif len(s_h) < len(h_r):
            s_h += ('',)
        if len(s_t) > len(t_r):
            t_r += ('',)
        elif len(s_t) < len(t_r):
            s_t += ('',)
        #replace head and tail by interweaving the substitution arrays and
        #split arrays
        peptide = (''.join([x for pair in zip(s_h, h_r) for x in pair]),
                   ''.join([x for pair in zip(s_t, t_r) for x in pair]))
    #edman failure
    #convert peptide head into list of cumulative gaps.
    #gaps is initialized as the sequence of gaps under ideal (p=1) conditions
    #stored as a list of pairs (gap, label). each gap is a cumulative gap from
    #the beginning of the peptide.
    #iterate through each item in head, adding a gap location to gaps whenever
    #a label acid with a window is found; index + 1 is used as the cumulative
    #distance because python indexing starts at 0. tuplefy to prevent bugs,
    #like the one below where I modified gaps in place while iterating over
    #them.
    gaps = tuple([(index + 1, acid) for index, acid in enumerate(peptide[0])
                  if acid in windows])
    #for each gap, add random error based on the bernoulli edman failure model.
    #implementation of this random delay is done by mapping all possibilities
    #onto the line segment [0,1] and choosing a random point on this segment
    #via random.random(). Then, the corresponding error is found by
    #accumulating the probabilities for successive errors starting with 0 until
    #the cumulative distribution surpasses the random point chosen.
    #bug found: in this function, i have several loops that iterate over gaps
    #and modify them at the same time. this is a major bug, and i need to learn
    #never to modify lists in place unless I really know what I'm doing. The
    #solution is to tuplefy gaps above to prevent bugs and copy it into a list.
    modified_gaps = list(gaps)
    #cumulative error to keep track of how big of cumulative e to add as the
    #array being modified is array of cumulative gaps
    cumulative_e = 0
    for index, gap in enumerate(gaps):
        #convert cumulative gap to difference gap for probability formula. it
        #is ok to use gaps instead of modified gaps here because every edman
        #failure gap is independent of the others
        d = (gaps[index][0]  - gaps[index - 1][0]
             if index > 0 else gaps[index][0])
        random_point = random.random()
        #delay
        e = 0
        #keeps track of cumulative probability of delays <= e
        accumulator = 0.0
        #this while True may take a long time if accumulator approaches 1.0
        #slowly and random_point is near 1. one solution based on empirical
        #evidence is to limit e to be under 1000 delays. d=50,p=0.97,it takes
        #only 20 cycles to have accumulator surpass 1.0 - 10**-14. Surpassing
        #1.0 - 10**-15 takes indefinitely long as the probability becomes below
        #machine precision. Alternate solution involves storing prior
        #accumulator value and seeing if difference becomes 0.0.
        prior_accumulator = -1.0
        while accumulator - prior_accumulator > 0.0:
            #store current value of accumulator to make sure change does not
            #drop below machine precision
            prior_accumulator = accumulator
            #bernoulli probability of total gap length being d + e, where d is
            #ideal gap length and e is delay
            accumulator += _dp(d, e, p)
            #use >= instead of > because random.random() generates on [0,1)
            if accumulator >= random_point:
                break
            e += 1
        #bug found: write unit test for bug where we add error in relation to
        #original location of gap, not the cumulative error over all gaps.
        #solution is to use cumulative_e to store total error to add to each
        #gap
        cumulative_e += e
        modified_gaps[index] = (gap[0] + cumulative_e, gap[1])
    #for every head fluor, generate random photobleaching time. a fluor being
    #photobleached due to a particular laser exposure means that while it was
    #visible during that laser exposure, it will not be visible during the next
    #laser exposure. hence, if a fluor photobleaches due to the laser exposre
    #directly before the k'th edman reaction, it will appear as though that
    #edman reaction caused the fluor to cleave. more than one color may
    #experience a luminosity drop simultaneously during the same edman
    #reaction.
    #update gaps to reflect the above edman failures. remember never to modify
    #list while iterating through it unless you are really sure
    gaps = tuple(modified_gaps)
    for index, gap in enumerate(gaps):
        random_point = random.random()
        #keeps track of cumulative probability of bleaching
        accumulator = 0.0
        #list of laser exposures prior to this position
        #bug found:create regression unit test for bug when this statement did
        #not have the if x < gap[0] and x - 1 < gap[0] conditions in it. This
        #lack of conditions caused each fluor to be exposed to more lasers than
        #necessary, causing unwarranted photobleaching
        exposures = sorted(list(set([x for x in windows[gap[1]]
                                     if x < gap[0] - 1] +
                                    [x - 1 for x in windows[gap[1]]
                                     if x - 1 < gap[0] - 1])))
        #survival indicates how many laser cycles the fluor survives. for
        #example, if there is a laser exposure between every edman cycle and an
        #initial exposure before any edman cycles, a fluor surviving 0 cycles
        #will yield a signal drop for the first edman cycle. a fluor with
        #survival = 3 will yield a signal for the fourth edman cycle. the
        #windowed works similarly. this implementation iterates over all
        #exposures to which the fluor can be exposed to via enumerate: survival
        #is referred to by the index in the list, and position is referred to
        #by the value. accumulator surpassing random_point during an iteration
        #means the signal drop will be observed during position + 1. once
        #observed, the loop breaks; no changes occur to those fluors that can
        #survive all exposures. the signal is recorded only if it occurs within
        #a window, i.e. between two consecutive exposures of its color.
        #otherwise, it is lost. it is assumed a fluor cannot photobleach
        #without at least one exposure.
        for survival, position in enumerate(exposures):
            accumulator += math.e**(-b * survival)
            #use >= instead of > because random.random() generates on [0,1)
            if accumulator  * (1 - math.e**-b) >= random_point:
                modified_gaps[index] = (position + 1, gap[1])
                break
    #for every tail fluor, generate random photobleaching time; similar
    #algorithm as above
    #list of all tail acids
    tail_acids = [acid for acid in windows
                  for x in range(peptide[1].count(acid))]
    for index, acid in enumerate(tail_acids):
        random_point = random.random()
        accumulator = 0.0
        exposures = sorted(list(set([x for x in windows[acid]] +
                                    [x - 1 for x in windows[acid]])))
        for survival, position in enumerate(exposures):
            accumulator += math.e**(-b * survival)
            #use >= instead of > because random.random() generates on [0,1)
            if accumulator * (1 - math.e**-b) >= random_point:
                modified_gaps.append((position + 1, acid))
                break
    #remove all gaps that are not bounded by two windows as they are
    #unobservable even if they photobleached or delayed into those locations.
    #TODO:is it possible to have algorithms based on longer windows than length
    #1? no need to construct list or sort exposures for purpose of checking if
    #gap is between two of them.
    filtered_gaps = []
    for acid in windows:
        exposures = set([x for x in windows[acid]] +
                        [x - 1 for x in windows[acid]])
        filtered_gaps.extend([gap for gap in modified_gaps
                              if (gap[1] == acid and
                                  gap[0] in exposures and
                                  gap[0] - 1 in exposures)])
    #remove identical entries
    gaps = list(set(filtered_gaps))
    #sort by position and return
    gaps = sorted(gaps, key=lambda gaps: gaps[0])
    return tuple(gaps)

def monte_carlo_dictionary(peptides, signals, p, b, u, windows,
                           sample_size=1000, result_queue=None, child_number=0,
                           silent=True):
    """
    Generate dictionary mapping protein quantities to signal quantities via
    Monte Carlo random sampling.

    Args:
        peptides: Dictionary of peptides used as signal sources. It is in the
            same format as the dictionary returned by attach or discard.
            {'PROTEIN 1': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             'PROTEIN 2': (('PEPTIDE HEAD 1', 'PEPTIDE TAIL 1'),
                           ('PEPTIDE HEAD 2', 'PEPTIDE TAIL 2'),
                            ...),
             ...}
        signals: Set of signals for which the matrix will be generated. Each
            signal is a tuple ((gap1, acid1), (gap2, acid2), ...). Gaps are
            cumulative. The gaps will first be windowed based on the windows
            argument based on window_filter().
        p: Probability of an Edman reaction succeeding. All Edman reactions are
            modelled as independent Bernoulli variables. p must be a number in
            [0, 1] inclusive.
        b: Photobleaching survival constant. Photobleaching is modeled as an
            exponential survival function s(k) = e^-bk, modeling the
            probability of a fluor surviving k laser exposures.
        windows: Laser excitations patterns for each color stored as a
            dictionary. Keys are single-letter strings representing the labeled
            acids, and values are lists or tuples representing the sequence of
            windows during which peptide luminosities are observed. Each member
            of the sequence refers to the number of the Edman cycle for which a
            difference in luminosity is searched for. For example, (3, 4, 7)
            would represent the search for luminosity drops due to the third,
            fourth, and seventh Edman cycle. This means there will be an
            exposure between the second and third Edman, the third and fourth
            Edman, between the fourth and fifth Edman, followed by exposures
            directly before and after the seventh Edman. If position 1 is
            indicated, there is an initial exposure before any Edmans. Thus,
            the dictionary format is e.g. {'E': (3, 4, 7), 'K': (7, 8, 9)}.
            Each amino acid is assumed to be labeled with a distinct color. If
            more than one amino acid is labeled with the same color,
            homogenize() needs to be applied. Having windows corresponding to
            large numbers of Edman reactions means reactions will keep being
            applied that long; the tails need to survive all of these.
        sample_size: Number of samples to take of each protein. If a sample
            generates an empty signal, it is still considered as taken, and it
            is not added to the monte carlo matrix.
        result_queue: Used by monte_carlo_dictionary_MP. If None, returns via
            return. If a multiprocessing.Queue is given, puts the results into
            it.
        child_number: Only affects progress outputs by assigning this instance
            of monte_carlo_dictionary a child number. This is useful if using
            edman_failure_gaps_MP.
        silent: Boolean indicating whether progress should be printed to
            standard output.

    Returns:
        Dictionary of samples generated via random_signal a la Monte Carlo.
        Format is {(signal tuple): {protein: quantity}}. The empty signal, if
        generated, is also stored.
    """
    raise DeprecationWarning
    #progress and chronometry; see comments below regarding using cumulative
    #timing
    interval_time = time.clock()
    cumulative_time = 0.0
    dictionary_progress = 0
    #ensure constants passed are floats
    p, b = float(p), float(b)
    #perform windowing on signals
    signals = set(window_filter(list(signals), windows))
    #cannot use the name monte_carlo_dictionary because it may conflict with
    #name of this function and cause bugs
    mc_dictionary = {}
    for protein in peptides:
        for i in xrange(sample_size * len(peptides[protein])):
            #choose a peptide at random for this protein
            peptide = random.choice([candidate
                                     for candidate in peptides[protein]])
            sample_signal = random_signal(peptide, p, b, u, windows)
            if sample_signal in signals:
                mc_dictionary.setdefault(sample_signal, {})
                mc_dictionary[sample_signal].setdefault(protein, 0)
                mc_dictionary[sample_signal][protein] += 1
        if not silent:
            dictionary_progress += 1
            #cumulative time chronometry should work unless a protein takes
            #more than ~30 min to run through all samples
            cumulative_time += time.clock() - interval_time
            interval_time = time.clock()
            if dictionary_progress > 0 and dictionary_progress % 100 == 0:
                #using time.clock() - starting_time is prone to underflow after
                #~30 min; this function is expected to take a while, so using
                #cumulative time
                print(str(dictionary_progress) + " of " + str(len(peptides)) +
                      " proteins completed in " + str(cumulative_time) +
                      ", average speed proteins/sec = " +
                      str(float(dictionary_progress) / cumulative_time))
    if result_queue is None:
        return (mc_dictionary, sample_size)
    else:
        result_queue.put((mc_dictionary, sample_size))

def monte_carlo_dictionary_MP(peptides, signals, p, b, windows,
                              sample_size=1000, silent=True):
    """
    Wrapper for running monte_carlo_dictionary on multiple processors.
    Functions identically to monte_carlo_dictionary; for usage, see
    documentation for the original single-processor variant.
    """
    raise DeprecationWarning
    child_count = multiprocessing.cpu_count()
    #spcb = samples per child base
    spcb = sample_size / child_count
    #spcr = proteins per child remainder
    spcr = sample_size % child_count
    child_sample_list = [spcb + 1] * spcr
    child_sample_list += [spcb] * (child_count - spcr)
    if not silent:
        print("number of samples: " + str(samples) + "; cpu count: " +
              str(child_count) + "; spawning children with " +
              str([len(x) for x in child_sample_list]) +
              " samples each")
    result_queue = multiprocessing.Queue()
    mc_dictionary = {}
    child_processes = []
    for c, child_samples in enumerate(child_sample_list):
        child_process = multiprocessing.Process(target=monte_carlo_dictionary,
                                                args=(peptides, signals, p, b,
                                                      windows, child_samples,
                                                      result_queue, c, silent))
        child_process.start()
        child_processes.append(child_process)
    results = []
    for child in child_processes:
        results.append(result_queue.get())
    for child in child_processes:
        child.join()
    for result in results:
        r_mc, r_ss = result
        for signal, proteins in r_mc.iteritems():
            mc_dictionary.setdefault(signal, {})
            for protein in proteins:
                mc_dictionary[signal].setdefault(protein, 0)
                mc_dictionary[signal][protein] += proteins[protein]
    return (mc_dictionary, sample_size)

class SignalTrie:
    """
    An implementation of a trie (prefix tree) structure to store large numbers
    of signals, and for each of these signals to track the number of times they
    were generated by particular souce proteins. The trie's root node is null,
    representing an empty signal. All descendant nodes identify themselves by a
    letter representing an amino acid and an integer for the length of the gap
    preceeding it. A node represents the signal composed by concatenating
    nodes' gaps and amino acids transversed to reach it, itself included. Each
    node contains a dictionary whose keys are (interval, amino acid) pairs
    pointing to further sequence members; each node is likewise pointed to by
    its ancestor. To count the number of times a protein generated the signal
    represented by the node, each node contains a dictionary mapping the source
    protein of its signal to an integer representing the number its signals
    from that protein.
    """
    def __init__(self, (pg, aa)):
        """
        Initializes this node to be made of amino acid aa with preceeding
        gap length pg. This node's descendant dictionary is empty, and it has
        no signal counts at initialization.
        """
        #(preceeding gap length, amino acid)
        self.signal_block = (pg, aa)
        #{next_signal_block: SignalTrie}
        self.descendants = {}
        #{protein: count}
        self.signal_count = {}
    def add_descendant(self, subsignal, source_protein):
        """
        Increments the count of the signal represented in the tree by this node
        followed by subsignal by one, with source_protein being the source of
        the signal. If the signal is not yet present in the tree, recursively
        adds subsignal as a descendant of this node.

        Args:
            subsignal: The remainder following this node of the signal to be
                incremented. It is a tuple as for all signals
                ((gap, aa), (gap, aa), ...).
            source_protein: String, name of protein that generated the signal.

        Returns:
            Self.
        """
        if len(subsignal) == 0:
            return
        elif self.signal_block == (None, None):
            self.descendants.setdefault(subsignal[0], SignalTrie(subsignal[0]))
            self.descendants[subsignal[0]].add_descendant(subsignal,
                                                          source_protein)
        elif len(subsignal) == 1:
            self.signal_count.setdefault(source_protein, 0)
            self.signal_count[source_protein] += 1
        else:
            self.descendants.setdefault(subsignal[1], SignalTrie(subsignal[1]))
            self.descendants[subsignal[1]].add_descendant(subsignal[1:],
                                                          source_protein)
        return self
    def set_descendant(self, subsignal, count):
        """
        Sets the signal_count in the leaf representing subsignal to a copy of
        count.

        Args:
            subsignal: The remainder following this node of the signal whose
                signal count is to be changed to count's copy. It is a tuple as
                for all signals
                ((gap, aa), (gap, aa), ...).
            count: Set subsignal's signal count to a copy of this dictionary.

        Returns:
            Self.
        """
        if len(subsignal) == 0:
            return
        elif self.signal_block == (None, None):
            self.descendants.setdefault(subsignal[0], SignalTrie(subsignal[0]))
            self.descendants[subsignal[0]].set_descendant(subsignal, count)
        elif len(subsignal) == 1:
            self.signal_count = count.copy()
        else:
            self.descendants.setdefault(subsignal[1], SignalTrie(subsignal[1]))
            self.descendants[subsignal[1]].set_descendant(subsignal[1:], count)
        return self
    def get_descendant(self, subsignal):
        """
        Return pointer to node represented by subsignal.

        Args:
            subsignal: The remainder following this node of the subsignal whose
                pointer to return. It is a tuple as for all signals
                ((gap, aa), (gap, aa), ...).

        Returns:
            Pointer to node represented by subsignal if it exists, None
                otherwise.
        """
        if len(subsignal) == 0:
            return
        elif self.signal_block == (None, None):
            if subsignal[0] in self.descendants:
                return self.descendants[subsignal[0]].get_descendant(subsignal)
            else:
                return None
        elif len(subsignal) == 1:
            return self
        else:
            if subsignal[1] in self.descendants:
                return\
                   self.descendants[subsignal[1]].get_descendant(subsignal[1:])
            else:
                return None
    def node_iterator(self):
        """
        An iterator over ALL descendant nodes and this node itself, whether
        with empty signal counts or not. See leaf_iterator to iterate over only
        nonempty descendant nodes. Iteration yields a tuple for each node:
        (signal represented by the node, signal counter, pointer to node)

        #??#This cannot be used while adding or removing nodes from the tree as
        #??#dictionaries cannot change size while iterating.
        """
        for d_trie in self.descendants.itervalues():
            for node in d_trie.node_iterator():
                if self.signal_block == (None, None):
                    yield node
                else:
                    #the decomposition is necessary to prepend
                    #self.signal_block
                    yield ((self.signal_block,) + node[0], node[1], node[2])
        #the following line is the critical difference between node_iterator
        #and leaf_iterator, as leaf_iterator is conditional on the node being
        #nonempty
        yield ((self.signal_block,), self.signal_count, self)
    def pop_node(self, prefix_signal=()):
        """
        Pops one terminal node and returns it and its signal. Cannot pop self.

        Args:
            prefix_signal: Tuple representing this node.

        Returns:
            (signal, node) where signal represents the node, and node has been
                popped.
        """
        d_gap, d_trie = self.descendants.items()[0]
        if len(d_trie.descendants) == 0:
            del self.descendants[d_gap]
            return prefix_signal + (d_gap,), d_trie
        else:
            return d_trie.pop_node(prefix_signal + (d_gap,))
    def leaf_iterator(self):
        """
        An iterator over all NONEMPTY descendant nodes and this node itself.
        See node_iterator to iterate over all descendant nodes whether empty or
        not. Iteration yields a tuple for each node:
        (signal represented by the node, signal counter, pointer to node)

        #??#This cannot be used while adding or removing nodes from the tree as
        #??#dictionaries cannot change size while iterating.
        """
        for d_trie in self.descendants.itervalues():
            for leaf in d_trie.leaf_iterator():
                if self.signal_block == (None, None):
                    yield leaf
                else:
                    #the decomposition is necessary to prepend
                    #self.signal_block
                    yield ((self.signal_block,) + leaf[0], leaf[1], leaf[2])
        #the following line is the critical difference between node_iterator
        #and leaf_iterator, as leaf_iterator is conditional on the node being
        #nonempty
        if len(self.signal_count) > 0:
            yield ((self.signal_block,), self.signal_count, self)
    def find_uniques(self, worst_ratio, absolute_min, maximum_secondary=None):
        """
        Returns all signals and their most responsible proteins where these
        protein are

        A. either unique (to force this requirement , set worst_ratio to None)
        OR ratio of the most responsible protein to the second most responsible
        protein is at least worst_ratio

        --AND--

        B. there are at least absolute_min signals from the most responsible
        protein

        Args:
            worst_ratio: Floating point ratio of protein that most frequently
                generated this signal to second most frequent protein source.
                If None, only those signals that are absolutely unique to a
                protein, i.e. there is only one protein that produced them,
                are returned. Otherwise, only those signals where the ratio
                exceeds the worst_ratio arguments are returned.
            absolute_min: Integer indicating for a signal to be returned it
                must have been generated at least an absolute_min number of
                times by the most responsible protein.
            maximum_secondary: If not None, integer indicating the maximum
                quantity of the second_worst protein.

        Returns:
            Dictionary of
                {signal:
           ((protein that most frequently generated this signal, its quantity),
             (tuples of proteins that are second most frequent source and their
              quantities), (total number of tertiary proteins)}
        """
        #this is an old implementation not using the iterators, but seems to
        #work fine
        uniques = {}
        #first see if self is a unique
        if len(self.signal_count) > 0:
            best = (None, 0)
            second = (None, 0)
            #find the best and second-best proteins
            for protein, count in self.signal_count.iteritems():
                if count > best[1]:
                    best = (protein, count)
                elif count > second[1]:
                    second = (protein, count)
            if (
                (best[1] >= absolute_min)

                and

                        (
                                   (worst_ratio is None and second[0] is None)
                                   or
                                   (worst_ratio is not None and second[1] == 0)
                                   or
                                   (worst_ratio is not None and
                                    float(best[1]) / second[1] >= worst_ratio)
                        )

                and

                        (
                                   maximum_secondary is None or
                                   second[0] is None or
                                   second[1] <= maximum_secondary
                        )
               ):
                #best protein, array of second-best protein,
                #total count of tertiary
                uniques.setdefault((self.signal_block,), [best, [second], 0])
                for protein, count in self.signal_count.iteritems():
                    if count == second[1] and protein != second[0]:
                        uniques[(self.signal_block,)][1].append((protein, count))
                    elif count < second[1]:
                        uniques[(self.signal_block,)][2] += count
        #now recursively check all descendants for uniques
        for block, descendant in self.descendants.iteritems():
            d_u = descendant.find_uniques(worst_ratio,
                                          absolute_min,
                                          maximum_secondary)
            for signal, (best, secondary, tertiary) in d_u.iteritems():
                if self.signal_block != (None, None):
                    uniques.setdefault((self.signal_block,) + signal,
                                       (best, secondary, tertiary))
                else:
                    uniques.setdefault(signal, (best, secondary, tertiary))
        return uniques
    def find_uniques_absolute(self, minimum_best, maximum_secondary):
        """
        Returns all signals and their most responsible and second most
        responsible proteins, as long as there are at least minimum_best
        primary proteins and at most maximum_secondary secondary proteins.
        """
        #this is an old implementation not using the iterators, but seems to
        #work fine
        uniques = {}
        #first see if self is a unique
        if len(self.signal_count) > 0:
            best = (None, 0)
            second = (None, 0)
            #find the best and second-best proteins
            for protein, count in self.signal_count.iteritems():
                if count > best[1]:
                    best = (protein, count)
                elif count > second[1]:
                    second = (protein, count)
            if best[1] >= minimum_best and second[1] <= maximum_secondary:
                #best protein, array of second-best protein,
                #total count of tertiary
                uniques.setdefault((self.signal_block,), [best, [second], 0])
                for protein, count in self.signal_count.iteritems():
                    if count == second[1] and protein != second[0]:
                        uniques[(self.signal_block,)][1].append((protein, count))
                    elif count < second[1]:
                        uniques[(self.signal_block,)][2] += count
                #coding error: uniques[self.signal_block][1] is a tuple, so can't
                #retuplefy second's array ???????????
                #uniques[self.signal_block][1] =\
                #                           tuple(uniques[self.signal_block][1])
        #now recursively check all descendants for uniques
        for block, descendant in self.descendants.iteritems():
            d_u = descendant.find_uniques_absolute(minimum_best,
                                                   maximum_secondary)
            #as above, best = (best protein, its quantity) and
            #second = (second best protein if present, its quantity)
            while len(d_u) > 0:
                signal, (best, second, tertiary) = d_u.popitem()
                if self.signal_block != (None, None):
                    uniques.setdefault((self.signal_block,) + signal,
                                       (best, second, tertiary))
                else:
                    uniques.setdefault(signal, (best, second, tertiary))
        return uniques
    def count_nodes(self):
        """
        Count total number of nodes in the trie rooted at this node.

        Returns:
            (Number of empty nodes, number of non-empty nodes)
        """
        empty, used = 0, 0
        for leaf in self.node_iterator():
            assert len(leaf[1]) >= 0
            if len(leaf[1]) == 0:
                empty += 1
            else:
                used += 1
        return empty, used
        #old implementation below not using leaf_iterator
        #empty = 0
        #used = 0
        #for block, descendant in self.descendants.iteritems():
        #    d_e, d_u = descendant.count_nodes()
        #    empty += d_e
        #    used += d_u
        #if len(self.signal_count) == 0:
        #    empty += 1
        #else:
        #    used += 1
        #return empty, used
    def prune(self, signal):
        """
        Return the signal back along with its signal counts. If the node
        representing the signal has no descendants, remove it from the trie. If
        it has descendants, reset its signal count dictionary to empty.

        Args:
            signal: Signal to remove from the trie. If it is not present, an
                AssertionError. If signal is a prefix of a longer signal, i.e.
                the prefix trie has non-empty descendant nodes from the
                terminal node of the signal given, then the node remains but
                its signal count dictionary is emptied.

        Returns:
            (signal, signal_count dictionary for the signal)
        """
        #all signals to be pruned must be non-empty
        assert len(signal) > 0
        #the recursion uses the parent node to prune its descendant, hence the
        #root (None, None) node must be the only one to prune signals length 1
        if len(signal) == 1:
            assert self.signal_block == (None, None)
        #if the signal is longer than 1 block long, and this block is the root
        #(None, None) node, then the first signal block must be a descendant of
        #the root node
        elif self.signal_block == (None, None):
            assert signal[0] in self.descendants
        #if the signal is longer than 1 block long, and this block is not the
        #root node, then the first signal block must refer to this node and its
        #second signal block must be a descendant
        else:
            assert signal[0] == self.signal_block,\
                   ('self.signal_block ' + str(self.signal_block) +
                    '; signal ' + str(signal))
            assert signal[1] in self.descendants
        #if the signal is length one, then it is just a length one signal that
        #is a direct descendant of the root (None, None) node; recursion would
        #not result in having a non-root node receiving a length one signal
        if len(signal) == 1:
            if len(self.descendants[signal[0]].descendants) == 0:
                #remove descendant node if it has no children of its own
                return (signal, self.descendants.pop(signal[0]).signal_count)
            else:
                #if desendant node has children of its own, then empty signal
                #count dictionary but leave the node itself
                s_c = self.descendants[signal[0]].signal_count
                self.descendants[signal[0]].signal_count = {}
                return (signal, s_c)
        #if the signal is longer than 1 block, and this node is the root (None,
        #None) node, then pass the recursion to the next node
        elif self.signal_block == (None, None):
            return self.descendants[signal[0]].prune(signal)
        #if the signal is longer than 1 block and this is not a root node, then
        #it must have had its function called from its parent. if the signal is
        #length 2, prune the child and return (base case); otherwise, recurse
        else:
            if len(signal) == 2:
                if len(self.descendants[signal[1]].descendants) == 0:
                    #remove descendant node if it has no children of its own
                    return (signal,
                            self.descendants.pop(signal[1]).signal_count)
                else:
                    #if desendant node has children of its own, then empty
                    #signal count dictionary but leave the node itself
                    s_c = self.descendants[signal[1]].signal_count
                    self.descendants[signal[1]].signal_count = {}
                    return (signal, s_c)
            else:
                r = self.descendants[signal[1]].prune(signal[1:])
                return ((self.signal_block,) + r[0], r[1])
    def graft(self, signal, signal_count):
        """
        Add a signal to this trie with given protein signal counts. If the
        signal is already present in the trie, add the protein signal counts.

        Args:
            signal: The signal sequence to add. This function is recursive, so
                the full signal to be added is the concatenation of the signal
                represented by the node called and subsignal passed.
            signal_count: If the signal is new, the signal_count dictionary
                passed will be used as its protein source count. If the signal
                exists, the signal counts passed will be added to it.

        Returns:
            Self.
        """
        #the signal passed cannot be empty
        assert len(signal) > 0
        #the signal's first block must either match this node (otherwise we
        #should not be here), or this is the root (None, None) node and the
        #first block is a descendant of the root
        assert signal[0] == self.signal_block or\
               self.signal_block == (None, None),\
               ('signal: ' + str(signal) +
                '; self.signal_block: ' + str(self.signal_block))
        #the signal_count must be non-zero, otherwise we will be adding empty
        #leaf nodes
        assert len(signal_count) > 0
        #if this is the root node, pass to descendant
        if self.signal_block == (None, None):
            self.descendants.setdefault(signal[0], SignalTrie(signal[0]))
            self.descendants[signal[0]].graft(signal, signal_count)
        #if this is not the root node, and the signal length is one, then we
        #must be referring to this node
        elif len(signal) == 1:
            for protein in signal_count:
                self.signal_count.setdefault(protein, 0)
                self.signal_count[protein] += signal_count[protein]
        #this is not the root node, and there is still signal to recurse
        #through
        else:
            self.descendants.setdefault(signal[1], SignalTrie(signal[1]))
            self.descendants[signal[1]].graft(signal[1:], signal_count)
        return self
    def merge(self, trie, cycles=None):
        """
        Iterates through all leafs in trie and grafts them onto this trie. This
        function can be used to copy an entire SignalTrie by initializing and
        emtpy trie via copy = SignalTrie((None, None)) and then
        copy.merge(original).

        Args:
            trie: The trie to merge with this one.
            cycles: If not None, merge only those leafs that are within cycles.

        Returns:
            Self.
        """
        #make sure this is called only from the root node
        assert self.signal_block == (None, None),\
               'merge can only be called on the root node'
        for leaf in trie.leaf_iterator():
            if cycles is None:
                self.graft(leaf[0], leaf[1])
            elif leaf[0][-1][0] <= cycles:
                self.graft(leaf[0], leaf[1])
        return self
    def truncating_projection(self, cycles):
        """
        This function projects the signals and their counts to signals that
        would be observed if the number of Edman cycles was truncated to a
        given number.

        Args:
            cycles: Number of Edman cycles to truncate to; integer.

        Returns:
            Self.
        """
        #first iterate through all leafs; for those leafs who require more than
        #'cycles' Edman cycles to observe, project them onto the shorter cycle
        #space and graft
        for leaf in self.leaf_iterator():
            #leaf[0][-1][0] is the number of Edman cycle at which last signal
            #drop was observed
            if leaf[0][-1][0] > cycles:
                #generate the projected signal; 's_b' is short for signal block
                projected_signal = tuple([s_b for s_b in leaf[0]
                                          if s_b[0] <= cycles])
                #this conditional necessary to prevent grafting empty
                #projected_signals. in this context, empty projected_signals
                #are those that are undetectable within 'cycles'
                if projected_signal:
                    self.graft(projected_signal, leaf[1])
        #now iterate and find all nodes and find those that fit within cycles
        #but point to descendants who do not.
        #note that the trie cannot be iterated and have nodes removed at the
        #same time, so it is necessary to break iterating and removing into two
        #steps
        terminal_node_pointers = [(node[2], descendant) #(term node, desc key)
         for node in self.node_iterator() for descendant in node[2].descendants
                        if node[0][-1][0] <= cycles and descendant[0] > cycles]
        #now iterate eliminate all the terminal node pointers
        for terminal_node, descendant_pointer in terminal_node_pointers:
            del terminal_node.descendants[descendant_pointer]
        #finally, we need to eliminate branches of nodes that do not lead to
        #leaves by iterating over all leaves and (special case) root node and
        #removing those descendants that have no leaves whatsoever to iterate
        #over
        terminal_leaf_pointers = []
        for leaf in self.leaf_iterator():
            for descendant, d_pointer in leaf[2].descendants.iteritems():
                has_subleaf = False
                for subleaf in d_pointer.leaf_iterator():
                    has_subleaf = True
                    break
                if not has_subleaf:
                    terminal_leaf_pointers.append((leaf[2], descendant))
        #special case of root node is because it is never a leaf and will never
        #be iterated over by leaf_iterator
        for descendant, d_pointer in self.descendants.iteritems():
            has_subleaf = False
            for subleaf in d_pointer.leaf_iterator():
                has_subleaf = True
                break
            if not has_subleaf:
                terminal_leaf_pointers.append((self, descendant))
        for leaf_pointer, descendant in terminal_leaf_pointers:
            del leaf_pointer.descendants[descendant]
        return self

class SlimSignalTrie:
    def __init__(self):
        self.descendants = {}
        self.proteins = set()
    def add_proteins(self, subsignal, proteins):
        self.descendants.setdefault(subsignal[0], SlimSignalTrie())
        if len(subsignal) > 1:
            self.descendants[subsignal[0]].add_proteins(subsignal[1:],
                                                        proteins)
        else:
            self.descendants[subsignal[0]].proteins |= proteins
    def get_proteins(self, subsignal):
        if len(subsignal) == 1:
            if subsignal[0] in self.descendants:
                return self.descendants[subsignal[0]].proteins
            else:
                return set()
        elif subsignal[0] in self.descendants:
            return self.descendants[subsignal[0]].get_proteins(subsignal[1:])
        else:
            return set()
    def compact_proteins(self, threshold=1):
        self.proteins = True if len(self.proteins) > threshold else False
        for s, n in self.descendants.iteritems():
            n.compact_proteins()

def monte_carlo_trie(peptides, p, b, u, windows, sample_size=100,
                      random_seed=random.random(), silent=True):
    """
    Generates sample_size number of random signals based on parameters given
    and returns them represented by SignalTrie.

    Returns:
        Returns a SignalTrie that represents all the signals generated by the
            peptides. The tree structure is a recursive
    """
    if not silent:
        print('monte_carlo_trie starting at ' + str(datetime.datetime.now()))
        sys.stdout.flush()
    #for tracking computational progress; total signals is total number of
    #signals that will be generated
    signal_progress = 0
    last_print = 0
    total_signals = (sum([len(attached)
                          for attached in peptides.itervalues()]) *
                     sample_size)
    update_interval = total_signals / 10
    #initialize the signal tree to be generated
    return_trie = SignalTrie((None, None))
    #initialize random number generator for this call
    random.seed(random_seed)
    #convert windows to randsiggen
    rsg_windows = {}
    for acid, positions in windows.iteritems():
        rsg_windows.setdefault(acid, max(positions))
    for protein in peptides:
        #weigh number of samples taken by how many peptides are yielded by
        #this protein; hence total number of peptide samples for this
        #protein is sample_size * len(peptides[protein])
        #generate array indicating how many of each peptide to sample
        for i, peptide in enumerate(peptides[protein]):
            sample_counter = sample_size
            while sample_counter > 0:
                batch_size = min(10**3, sample_counter)
                #batch = smlr.random_signal_batch(peptide, p, b, u, windows,
                #                                 batch_size)
                randsiggen.random_signal(peptide, protein, p, b, u,
                                         rsg_windows, batch_size,
                                         random.randint(0, 10**8), return_trie)
                #for signal in batch:
                #    #sorting required as cython balks at lamda sorting
                #    return_trie.add_descendant(
                #                            sorted(signal, key=lambda x: x[0]),
                #                            protein)
                sample_counter -= batch_size
                signal_progress += batch_size
                #for signal in batch:
                #    del signal
                #del batch
                #gc.collect()
            if (not silent and
                (signal_progress - last_print >= update_interval)):
                print(str(signal_progress) + ' of ' + str(total_signals) +
                      ' signals generated by ' + str(datetime.datetime.now()))
                #print('return_trie reference count ' +
                #       str(sys.getrefcount(return_trie)))
                last_print = signal_progress
                sys.stdout.flush()
    return return_trie

def monte_carlo_trie_MP(peptides, p, b, u, windows, sample_size=1000,
                        alt_sample_sizes=None, child_count=None, silent=True):
    raise DeprecationWarning
    assert alt_sample_sizes == None, 'alt_sample_sizes not implemented'
    child_count = (multiprocessing.cpu_count() if child_count is None
                   else min(child_count, len(peptides)))
    child_protein_list = _split_peptides_for_mp(peptides, child_count)
    if not silent:
        print("number of proteins: " + str(len(peptides)) + "; child count: " +
              str(child_count) + "; spawning children with " +
              str([len(x) for x in child_protein_list]) +
              " proteins each")
        sys.stdout.flush()
    result_queue = multiprocessing.Queue()
    child_processes = []
    def random_signal_multiplexer(child_peptides, p, b, u, windows,
                                  child_number=None):
        signal_progress = 0
        last_print = 0
        total_signals = (sum([len(peptides)
                              for peptides in child_peptides.itervalues()]) *
                         sample_size)
        if not silent:
            print('child number ' + str(child_number)  + ' starting at ' +
                  str(datetime.datetime.now()))
        for protein in child_peptides:
            #weigh number of samples taken by how many peptides are yielded by
            #this protein; hence total number of peptide samples for this
            #protein is sample_size * len(child_peptides[protein])
            for i, peptide in enumerate(child_peptides[protein]):
                sample_counter = sample_size
                while sample_counter > 0:
                    batch_size = min(10**4, sample_counter)
                    batch = smlr.random_signal_batch(peptide, p, b, u, windows,
                                                     batch_size)
                    for signal in batch:
                        #sorting required as cython balks at lamda sorting
                        result_queue.put((sorted(signal, key=lambda x: x[0]),
                                          protein))
                    sample_counter -= batch_size
                    signal_progress += batch_size
            if (signal_progress - last_print >= 10**6 or
                signal_progress % 10**6 == 0):
                    print('child number ' + str(child_number) + ': ' +
                          str(signal_progress) + ' of ' +
                          str(total_signals) + ' signals generated by ' +
                          str(datetime.datetime.now()) + ' sec')
                    last_print = signal_progress
                    sys.stdout.flush()
                #traditional python random_signal
                #for x in range(samples[i]):
                #   result_queue.put((random_signal(peptide, p, b, u, windows),
                #                     protein))
    for c, child_proteins in enumerate(child_protein_list):
        #TODO:python 2.6 cannot handle dictionary comprehensions; reverting to
        #setdefault
        #child_peptides = {protein: peptides[protein]
        #                  for protein in child_proteins}
        child_peptides = {}
        for protein in child_proteins:
            child_peptides.setdefault(protein, peptides[protein])
        #TODO:end reversion
        child_process = multiprocessing.Process(
                                    target=random_signal_multiplexer,
                                    args=(child_peptides, p, b, u, windows, c))
        child_process.start()
        child_processes.append(child_process)
    return_trie = SignalTrie((None, None))
    for c, child in enumerate(child_processes):
        for protein in child_protein_list[c]:
            for i in range(sample_size * len(peptides[protein])):
                signal, protein = result_queue.get()
                return_trie.add_descendant(signal, protein)
    for child in child_processes:
        child.join()
    return return_trie


class PolyfluorSignal:
    """
    A PolyfluorSignal represents the simulated fluorosequence of a
    PolyfluorPeptide as a tuple:

    ((amino_acid_1,
      position,
      frozenset(('u', boolean), ('p', delay), ('b', cycle))),

      ...

    )

    Each member of the tuple is a 3-tuple representing the removal of an amino
    acid. The first member of the 3-tuple is the amino-acid species string
    representation (e.g. lysine is 'K'), the second member is the _observed_
    position of the amino acid (i.e. after how many cycles it was removed, NOT
    necessarily its original position), and the third member is a frozenset
    carrying some information about the history of that fluorophore (i.e. the
    sequence of experimental events that resulted in the observed position
    being what it is). The frozenset members are themselves tuples, with the
    first member naming the type of experimental error/event that has taken
    place. For example, 'u' indicates that the fluor was dead or failed to
    attach, 'p' indicates some number of Edman failures in front of the amino
    acid, and 'b' indicates some photobleaching event. Not all events need to
    be included, in fact the frozenset can be empty. Furthermore, additional
    event types can be incorporated aside from the ones described here; the
    frozenset doesn't care what the content of the tuples is, and it is up to
    the class user to keep things tidy. The second (and optionally third or
    more) members of the error tuples describe or quantify the error. Usually,
    'u' (dead fluors) will be described by a boolean True, 'p' (Edman failures)
    will be described by an integer indicating the number of Edman failures
    ahead of the amino acid, and 'b' will be described by the integer number of
    cycles at which the photobleaching event took place.

    Attributes:
        peptide: Source PolyfluorPeptide for this signal.
        signal: The tuple representation of the fluorosequence as described
            above.
    """
    #frozenset(mutabledict.items())
    def __init__(self, peptide, signal=None):
        self.peptide = peptide
        if signal is None:
            self.signal = ()

    def default_simulation(self, num_cycles, p=1.0, b=0.0, u=0.0,
                           random_seed=None, num_mocks=0,
                           adjust_by_mocks=False, p2=None, b2=None):
        """
        Simulates fluorosequencing on this peptide using the model in [DOI:
        10.1371/journal.pcbi.1004080] and updates self.signal to reflect this
        simulation instance. See random_signal() above or randsiggen.c for a
        well-commented version of this algorithm. The difference is that this
        version tracks the changes using the error entries of the
        PolyfluorSignal.
        """
        if random_seed is None:
            random.seed()
        else:
            random.seed(random_seed)
        p, b, u = float(p), float(b), float(u)
        if p2 is not None:
            raise NotImplementedError
            p2r, p2p = p2
        if b2 is not None:
            b2r, b2p = b2
        #initialize fluorosequence to ideal configuration.
        #          aa     pos    err
        signal = tuple((aa[0], aa[1], []) for aa in self.peptide.peptide)
        #remove fluors based on u
        modified_signal = [(s[0], -1, [('u', True)])
                           if random.random() <= u else s
                           for s in signal]
        #place all dead fluors at the beginning
        modified_signal = sorted(modified_signal, key=lambda x:x[1])
        #TODO
        #TODO
        #TODO  NOTE THAT WHEN CORRELATING DEAD FLUORS WITH DATA TO ACCOUNT FOR
        #TODO  NOTE THE ('A', -1) SIGNALS BEING IN THE DICTIONARY KEYS BUT NOT
        #TODO  NOTE REALLY BEING PART OF THE SIGNAL
        #TODO
        #simulate photobleaching during mocks
        updated_signal = [x for x in modified_signal]
        for index, (aa, pos, err) in tuple(enumerate(modified_signal)):
            if pos == -1:
                continue
            random_point = random.random()
            accumulator = 0.0
            exposures = num_mocks
            for x in range(exposures):
                if b2 is None:
                    accumulator += math.e**(-b * x)
                    if accumulator * (1.0 - math.e**-b) >= random_point:
                        updated_signal[index] = (aa,
                                                 -2,
                                                 err + [('mb', x + 1)])
                        break
                else:
                    if x == b2p - 1:
                        accumulator += math.e**(-b * x)
                        if accumulator * (1.0 - math.e**-b2r) >= random_point:
                            updated_signal[index] = (aa,
                                                     -2,
                                                     err + [('mb', x + 1)])
                            break
                    elif x < b2p:
                        accumulator += math.e**(-b * x)
                        if accumulator * (1.0 - math.e**-b) >= random_point:
                            updated_signal[index] = (aa,
                                                     -2,
                                                     err + [('mb', x + 1)])
                            break
                    else:
                        accumulator += math.e**(-b2r * x)
                        if accumulator * (1.0 - math.e**-b2r) >= random_point:
                            updated_signal[index] = (aa,
                                                     -2,
                                                     err + [('mb', x + 1)])
                            break
        modified_signal = updated_signal
        #place all photobleached fluors at the beginning
        modified_signal = sorted(modified_signal, key=lambda x:x[1])
        #simulate Edman failure
        updated_signal = [x for x in modified_signal]
        cumulative_e = 0
        for index, (aa, pos, err) in tuple(enumerate(modified_signal)):
            if pos == -1 or pos == -2:
                continue
            d = (modified_signal[index][1] - modified_signal[index - 1][1]
                 if index > 0 else modified_signal[index][1])
            random_point = random.random()
            e = 0
            accumulator = 0.0
            prior_accumulator = -1.0
            #special case: if p is very low or 0, _dp(d, e, p) will return
            #extremely small values, causing an almost immediate exit out of
            #the loop due to accumulator - prior_accumulator = 0. This criteria
            #was made to deal with finite floating point precision. What should
            #happen is that essentially the fluor will never be sequenced.
            if p < 0.0001:
                e += 10 * num_cycles #This should place it well beyond the end
            #Another special case is if p is almost 1.0, this results in the
            #same problem as the above special case. We just assume perfect
            #sequencing
            elif p > 0.9999:
                pass
            else:
                #Before special case was added, this was the original loop.
                while accumulator - prior_accumulator > 0.0:
                    prior_accumulator = accumulator
                    accumulator += _dp(d, e, p)
                    if accumulator >= random_point:
                        break
                    e += 1
            cumulative_e += e
            updated_signal[index] = (aa,
                                     pos + cumulative_e + num_mocks,
                                     err + [('p', cumulative_e)])
        modified_signal = updated_signal
        #simulate photobleaching during edmans
        updated_signal = [x for x in modified_signal]
        for index, (aa, pos, err) in tuple(enumerate(modified_signal)):
            if pos == -1 or pos == -2:
                continue
            random_point = random.random()
            accumulator = 0.0
            exposures = min(num_cycles + 1, pos - num_mocks)
            for x in range(exposures):
                if b2 is None:
                    accumulator += math.e**(-b * x)
                    if accumulator * (1.0 - math.e**-b) >= random_point:
                        updated_signal[index] = (aa,
                                                 x + 1 + num_mocks,
                                                 err + [('b', x + 1)])
                        break
                else:
                    if x == b2p - 1:
                        accumulator += math.e**(-b * x)
                        if accumulator * (1.0 - math.e**-b2r) >= random_point:
                            updated_signal[index] = (aa,
                                                     x + 1 + num_mocks,
                                                     err + [('b', x + 1)])
                            break
                    elif x < b2p:
                        accumulator += math.e**(-b * x)
                        if accumulator * (1.0 - math.e**-b) >= random_point:
                            updated_signal[index] = (aa,
                                                     x + 1 + num_mocks,
                                                     err + [('b', x + 1)])
                            break
                    else:
                        accumulator += math.e**(-b2r * x)
                        if accumulator * (1.0 - math.e**-b2r) >= random_point:
                            updated_signal[index] = (aa,
                                                     x + 1 + num_mocks,
                                                     err + [('b', x + 1)])
                            break
        modified_signal = updated_signal
        #revert mock photobleached fluors to their original positions
        updated_signal = [x for x in modified_signal]
        for index, (aa, pos, err) in tuple(enumerate(modified_signal)):
            if pos == -2:
                fp = None
                for et, ep in err:
                    if et == 'mb':
                        fp = ep
                        break
                assert fp is not None
                updated_signal[index] = (aa, fp, err)
        modified_signal = updated_signal
        #sort by final position
        modified_signal = sorted(modified_signal, key=lambda x:x[1])
        #remove all positions after num_cycles + mocks
        modified_signal = [(aa, pos, err)
                           for (aa, pos, err) in modified_signal
                           if pos <= num_cycles + num_mocks]
        if adjust_by_mocks:
            raise NotImplementedError
            adjusted_signal = []
            for aa, pos, err in modified_signal:
                new_pos = pos + num_mocks
                new_err = frozenset([(t, p + num_mocks) for t, p in list(err)])
                adjusted_signal.append((aa, new_pos, new_err))
        return tuple((aa, pos, frozenset(err))
                     if err is not None else (aa, pos, frozenset())
                     for aa, pos, err in modified_signal)

    @staticmethod
    def strip_errors(signal):
        return (tuple((aa, pos) for aa, pos, err in signal),
                tuple(err for err in signal))

    def simulation_v2(
                      self,
                      num_cycles,
                      p,
                      b,
                      u,
                      random_seed=None,
                      num_mocks=0,
                     ):
        """
        A more flexible simulation framework based on tracking underlying
        peptide state during each experimental cycle.

        Lower-case x is reserved. Do not use to represent labelable amino
        acids.
        """
        raise NotImplementedError()
        if random_seed is None:
            random.seed()
        else:
            random.seed(random_seed)
        #Initialize list representation of physical peptide state
        last_dye_position = self.peptide.peptide[-1][1]
        amino_acid_set = set([aa for aa, pos in self.peptide.peptide])
        if 'x' in amino_acid_set:
            raise Exception("Function assumes lower-case x is reserved for "
                            "unlabeled.")
        molecule = ['x' for x in range(last_dye_position)]
        for aa, pos in self.peptide.peptide:
            molecule[pos] = aa
        #Initialize observed signal
        signal = []
        #Initialize error tracker
        errors = []
        #define steps/errors that are applied to the molecule
        def mock(molecule):
            updated_molecule = molecule
            signal_update = []
            errors_update = []
            return updated_molecule, signal_update, errors_update
        def edman(molecule):
            random_point = random.random()
            if random_point < p:
                if molecule[0] != 'x':
                    signal_update = [molecule[0]]
                else:
                    signal_update = []
                updated_molecule = molecule[1:]
                errors_update = []
            else:
                updated_molecule = molecule
                signal_update = []
                errors_update = ['p']
            return updated_molecule, signal_update, errors_update
        #Convert b into a per-cycle rate. Probability[photobleach]
        per_cycle_b = math.e**-b
        def exposure(molecule):
            random_point = random.point()
            #if random_point

 
class PolyfluorSignalTrie:
    """
    An implementation of a trie (prefix tree) to store large numbers of
    PolyfluorSignals, and for each of these signals to track the number of
    times they were generated by particular source proteins as well as the
    combination of events led to that signal.

    It is similar to SignalTrie except it has additional branching based on the
    error frozensets in PolyfluorSignals.
    """
    def __init__(self, (aa, pos, err)):
        self.signal_block = (aa, pos, err)
        self.descendants = {}
        self.signal_count = {}

    def add_descendant(self, subsignal, source_protein):
        if len(subsignal) == 0:
            return
        elif self.signal_block == (None, None, None):
            self.descendants.setdefault(subsignal[0],
                                        PolyfluorSignalTrie(subsignal[0]))
            self.descendants[subsignal[0]].add_descendant(subsignal,
                                                          source_protein)
        elif len(subsignal) == 1:
            self.signal_count.setdefault(source_protein, 0)
            self.signal_count[source_protein] += 1
        else:
            self.descendants.setdefault(subsignal[1],
                                        PolyfluorSignalTrie(subsignal[1]))
            self.descendants[subsignal[1]].add_descendant(subsignal[1:],
                                                          source_protein)
        return self

    def get_descendant(self, subsignal):
        if len(subsignal) == 0:
            return
        elif self.signal_block == (None, None, None):
            if subsignal[0] in self.descendants:
                return self.descendants[subsignal[0]].get_descendant(subsignal)
            else:
                return None
        elif len(subsignal) == 1:
            return self
        else:
            if subsignal[1] in self.descendants:
                return \
                   self.descendants[subsignal[1]].get_descendant(subsignal[1:])
            else:
                return None
        raise Exception

    def isoerr_get_descendant(self, subsignal):
        if len(subsignal) == 0:
            return
        subsignal = [s[:2] for s in subsignal]

    def graft(self, signal, signal_count):
        if self.signal_block == (None, None, None):
            self.descendants.setdefault(signal[0],
                                        PolyfluorSignalTrie(signal[0]))
            self.descendants[signal[0]].graft(signal, signal_count)
        elif len(signal) == 1:
            for protein in signal_count:
                self.signal_count.setdefault(protein, 0)
                self.signal_count[protein] += signal_count[protein]
        else:
            self.descendants.setdefault(signal[1],
                                        PolyfluorSignalTrie(signal[1]))
            self.descendants[signal[1]].graft(signal[1:], signal_count)
        return self

    def leaf_iterator(self):
        for d_trie in self.descendants.itervalues():
            for leaf in d_trie.leaf_iterator():
                if self.signal_block == (None, None, None):
                    yield leaf
                else:
                    yield((self.signal_block,) + leaf[0], leaf[1], leaf[2])
        if len(self.signal_count) > 0:
            yield ((self.signal_block,), self.signal_count, self)

    def merge(self, trie):
        if self.signal_block != (None, None, None):
            raise Exception("merge can only be called on root node.")
        for leaf in trie.leaf_iterator():
            self.graft(leaf[0])
        return self


class PolyfluorPeptide:
    """
    A PolyfluorPeptide represents a multiply-labeled peptide using a tuple:

    ((amino_acid_1, position_1), (amino_acid_2, position_2), ... )

    where amino_acid_i is the string representation of a labeled amino acid
    species (e.g. lysine is 'K'), and position_i is its position in the peptide
    (with the first position being 1). Only labeled amino acids are included in
    the tuple; all included amino acids are interpreted as labeled. An empty
    tuple indicates this peptide has no labelable amino acids.

    Attributes:
        parent_protein: String name of the parent protein.
        peptide: The tuple representation of the peptide as described above.
    """
    @staticmethod
    def sequence_to_peptide(sequence, acids=None):
        return tuple((acid, index + 1)
                     for index, acid in enumerate(sequence)
                     if acid in acids)

    @staticmethod
    def proteome_to_peptides(proteome, acids=None):
        return {protein:
                PolyfluorPeptide.sequence_to_peptide(sequence=sequence,
                                                     acids=acids)
                for protein, sequence in proteome.iteritems()}

    def __init__(self, parent_protein=None, sequence=None, acids=None,
                 peptide=None):
        if parent_protein is None:
            self.parent_protein = ''
        else:
            self.parent_protein = parent_protein
        if sequence is None:
            if peptide is None:
                self.peptide = ()
            else:
                self.peptide = peptide
        else:
            self.peptide = PolyfluorPeptide.sequence_to_peptide(sequence,
                                                                acids)

    def default_simulation(self, num_cycles, p=1.0, b=0.0, u=0.0, num_sims=1,
                           num_mocks=0, adjust_by_mocks=False, p2=None,
                           b2=None):
        signal = PolyfluorSignal(peptide=self, signal=None)
        return tuple(signal.default_simulation(num_cycles=num_cycles,
                                               p=p, b=b, u=u,
                                               random_seed=None,
                                               num_mocks=num_mocks,
                                               adjust_by_mocks=adjust_by_mocks,
                                               p2=p2, b2=b2)
                     for n in range(num_sims))

    def default_simulation_as_trie(self, num_cycles,
                                   p=1.0, b=0.0, u=0.0, num_sims=1, p2=None,
                                   b2=None):
        signal = PolyfluorSignal(peptide=self, signal=None)
        result = PolyfluorSignalTrie((None, None, None))
        for n in range(num_sims):
            s = signal.default_simulation(num_cycles=num_cycles,
                                          p=p, b=b, u=u,
                                          random_seed=None, p2=p2, b2=b2)
            result.add_descendant(s, self.parent_protein)
        return result

    def default_simulation_as_dict(self, num_cycles, p=1.0, b=0.0, u=0.0,
                                   num_sims=1, num_mocks=0,
                                   adjust_by_mocks=False, p2=None, b2=None):
        signal = PolyfluorSignal(peptide=self, signal=None)
        fluorosequences = (signal.default_simulation(num_cycles=num_cycles,
                                                     p=p, b=b, u=u,
                                                     random_seed=None,
                                                     num_mocks=num_mocks,
                                                     adjust_by_mocks=
                                                               adjust_by_mocks,
                                                     p2=p2, b2=b2)
                           for n in range(num_sims))
        d = {}
        for seq in fluorosequences:
            stripped_seq, stripped_err = PolyfluorSignal.strip_errors(seq)
            d.setdefault(stripped_seq, {}).setdefault(stripped_err, 0)
            d[stripped_seq][stripped_err] += 1
        return d


class PolyfluorPeptide_v2:
    @staticmethod
    def _define_reserved_character(sequence, labels):
        sequence_characters = set([L for L in sequence])
        characters_used = labels | sequence_characters
        possible_characters = (set([L for L in letters])
                               | set([d for d in digits]))
        characters_available = possible_characters - characters_used
        if len(characters_available) == 0:
            raise ValueError("sequence and labels use all possible "
                             "string.letters and string.digits. At least one "
                             "must remain available as a reserved letter for "
                             "this class.")
        return characters_available.pop()

    FluorEvent = namedtuple('FluorEvent',
                            [
                             'original_position',
                             'original_amino_acid',
                             'event',
                             'cycle_number',
                            ]
                           )

    def __init__(self, sequence, labels, parent_protein=None):
        """labels is a string, labels is a set"""
        self.molecule = tuple(enumerate(sequence, start=1))
        self.labels = labels
        if parent_protein is None:
            self.parent_protein = ''
        else:
            self.parent_protein = parent_protein
        self.reserved_character = \
               PolyfluorPeptide_v2._define_reserved_character(sequence, labels)

    def _mock(self, molecule, signal, history, removal_buffer, cycle_number,
              **experimental_parameters):
        pass

    def _edman(self, molecule, signal, history, removal_buffer, cycle_number,
               **experimental_parameters):
        if len(molecule) > 0:
            nterm_position, nterm_amino_acid = molecule[0]
            random_point = random.random()
            if random_point < experimental_parameters['p']:
                if nterm_amino_acid in self.labels:
                    emission = PolyfluorPeptide_v2.FluorEvent(
                                          original_position=nterm_position,
                                          original_amino_acid=nterm_amino_acid,
                                          event='edman',
                                          cycle_number=cycle_number,
                                                             )
                    removal_buffer.append(emission)
                    molecule.pop(0)
                else:
                    molecule.pop(0)
            else:
                error = FluorEvent(
                                   original_position=nterm_position,
                                   original_amino_acid=nterm_amino_acid,
                                   event='edman error',
                                   cycle_number=cycle_number,
                                  )
                history.append(error)
        else:
            pass

    def _tirf(self, molecule, signal, history, removal_buffer, cycle_number,
              **experimental_parameters):
        """Photobleaching events are assumed to occur during an exposure."""
        per_cycle_b = experimental_parameters.get('per_cycle_b',
                                         math.e**-experimental_parameters['b'])
        for i, (position, amino_acid) in enumerate(molecule):
            random_point = random.random()
            if random_point > per_cycle_b:
                emission = PolyfluorPeptide_v2.FluorEvent(
                                      original_position=position,
                                      original_amino_acid=amino_acid,
                                      event='dye destruction',
                                      cycle_number=cycle_number,
                                     )
                removal_buffer.append(emission)
                molecule[i] = self.reserved_character
        while removal_buffer:
            event = removal_buffer.pop()
            history.append(event)
            signal.append(event)

    def _dud(self, molecule, signal, history, removal_buffer, cycle_number,
             **experimental_parameters):
        for i, (position, amino_acid) in enumerate(molecule):
            random_point = random.random()
            if random_point < experimental_parameters['u']:
                error = PolyfluorPeptide_v2.FluorEvent(
                                   original_position=position,
                                   original_amino_acid=amino_acid,
                                   event='dye dud',
                                   cycle_number=cycle_number,
                                  )
                history.append(error)
                molecule[i] = self.reserved_character

    def simulate_type1(self, num_mocks, num_edmans, random_seed=None,
                       **experimental_parameters):
        """Assumes C-term anchoring."""
        if random_seed is None:
            random.seed()
        else:
            random.seed(random_seed)
        #The following five track the state of the simulation
        molecule = list(self.molecule)
        signal = []
        history = []
        removal_buffer = []
        cycle_number = 0
        self._dud(molecule, signal, history, removal_buffer, cycle_number,
                  **experimental_parameters)
        for mock in range(num_mocks):
            self._tirf(molecule, signal, history, removal_buffer, cycle_number,
                       **experimental_parameters)
            self._mock(molecule, signal, history, removal_buffer, cycle_number,
                       **experimental_parameters)
            cycle_number += 1
        for edman in range(num_edmans):
            self._tirf(molecule, signal, history, removal_buffer, cycle_number,
                       **experimental_parameters)
            self._edman(molecule, signal, history, removal_buffer, cycle_number,
                        **experimental_parameters)
            cycle_number += 1
        self._tirf(molecule, signal, history, removal_buffer, cycle_number,
                   **experimental_parameters)
        return molecule, signal, history, removal_buffer, cycle_number


def read_track_photometries_csv(path, downstep_filtered=False, head_truncate=0,
                                tail_truncate=0, omit_header=True,
                                channels=None):
    reader = csv.reader(open(path))
    #CHANNEL,FIELD,H,W,CATEGORY,FRAME 0,FRAME 1,FRAME 2,FRAME 3,FRAME 4, ...
    d = {}
    d2 = {}
    for r, row in enumerate(reader):
        if r == 0 and omit_header:
            continue
        head, frames = row[:5], row[5:]
        channel, field, h, w, category = head
        if channels is not None and channel not in channels:
            continue
        if h == 'None' or w == 'None':
            continue
        field, h, w = (int(round(float(field))),
                       int(round(float(h))),
                       int(round(float(w))))
        category = category[1:-1]
        category = category.split(' ')
        parsed_cat = tuple([True if c == 'True,' or c == 'True' else False
                            for c in category])
        if tail_truncate > 0:
            parsed_cat = parsed_cat[head_truncate:-1 * tail_truncate]
        else:
            parsed_cat = parsed_cat[head_truncate:]
        parsed_cat = tuple(parsed_cat)
        if downstep_filtered:
            if not (tuple(sorted(parsed_cat, reverse=True)) == parsed_cat and
                    parsed_cat[0]):
                continue
        parsed_frames = [int(round(float(f))) for f in frames]
        if tail_truncate > 0:
            parsed_frames = parsed_frames[head_truncate:-1 * tail_truncate]
        else:
            parsed_frames = parsed_frames[head_truncate:]
        parsed_frames = tuple(parsed_frames)
        d.setdefault(channel, {}).setdefault(field, {}).setdefault((h, w),
                                                (parsed_cat, parsed_frames, r))
        d2.setdefault(r, (channel, field, h, w, parsed_cat, parsed_frames))
    return d, d2


def _pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def _r_2(a, b):
    """a is data, b is fit"""
    a, b = np.array(a), np.array(b)
    return 1.0 - sum((a - b)**2) / sum((a - np.mean(a))**2)


def _check_no_downsteps(plateaus):
    if any([p1[0] < p2[0] for p1, p2 in _pairwise(plateaus)]):
        return False
    else:
        return True


def _plateau_fit(intensities, max_num_drops,
                 include_original_intensities=False, downsteps_only=False,
                 use_adjusted_r_2=False, delta_r_2=0.05,
                 original_intensities_only=True, **kwargs):
    best_fit, best_r_2, best_adj_r_2 = None, -1, -1
    #special case when all intensities uniform
    if len(set(intensities)) == 1:
        best_fit, best_r_2, best_adjusted_r2 = [[x for x in intensities]], 1.0, 1.0
    else:
        for drops in itertools.product(range(len(intensities)),
                                       repeat=max_num_drops):
            drops = sorted(list(set(drops)))
            drop_ends = [d - 1 for d in drops] + [len(intensities) - 1]
            if drop_ends[0] < 0:
                drop_ends = drop_ends[1:]
            else:
                drops.insert(0, 0)
            plateau_tuples = list(zip(drops, drop_ends))
            plateaus = [intensities[start:stop + 1]
                        for start, stop in plateau_tuples]
            plateau_fits = [[np.mean(plateau)] * len(plateau)
                            for plateau in plateaus]
            merged_fit = list(itertools.chain(*plateau_fits))
            r_2 = _r_2(intensities, merged_fit)
            if np.isnan(r_2):
                continue
            if downsteps_only and not _check_no_downsteps(plateau_fits):
                continue
            if use_adjusted_r_2:
                k = 2.0 * len(plateau_fits) - 1.0
                adj_r_2 = (1.0 -
                           (1.0 - r_2) * (len(intensities) - 1.0) /
                           (len(intensities) - k - 1.0))
                if best_fit is None or len(plateau_fits) <= len(best_fit):
                    if adj_r_2 > best_adj_r_2:
                        best_fit = plateau_fits
                        best_adj_r_2 = adj_r_2
                elif len(plateau_fits) > len(best_fit):
                    if adj_r_2 > best_adj_r_2 + delta_r_2:
                        best_fit = plateau_fits
                        best_adj_r_2 = adj_r_2
            else:
                if best_fit is None or len(plateau_fits) <= len(best_fit):
                    if r_2 > best_r_2:
                        best_fit = plateau_fits
                        best_r_2 = r_2
                elif len(plateau_fits) > len(best_fit):
                    if r_2 > best_r_2 + delta_r_2:
                        best_fit = plateau_fits
                        best_r_2 = r_2
        #if use_adjusted_r_2:
        #    assert best_adj_r_2 >= 0
        #else:
        #    assert best_r_2 >= 0
    if include_original_intensities and original_intensities_only:
        raise Exception
    if include_original_intensities:
        i = 0
        best_fit_originals = []
        for plateau in best_fit:
            best_fit_originals.append([])
            for v in plateau:
                best_fit_originals[-1].append((v, intensities[i]))
                i += 1
        best_fit = best_fit_originals
    elif original_intensities_only:
        i = 0
        best_fit_originals = []
        for plateau in best_fit:
            best_fit_originals.append([])
            for v in plateau:
                best_fit_originals[-1].append(intensities[i])
                i += 1
        best_fit = best_fit_originals
    if use_adjusted_r_2:
        best_r_2 = best_adj_r_2
    return best_fit, best_r_2


def _all_plateau_fits(intensities, max_num_drops, storage_r_2_cutoff=0.7):
    all_fits = []
    if len(set(intensities)) == 1:
        fit, r_2, adj_r2 = [[x for x in intensities]], 1.0, 1.0
        #merge original intensities into plateaus
        i = 0
        plateau_fits_originals = []
        for plateau in plateau_fits:
            plateau_fits_originals.append([])
            for v in plateau:
                plateau_fits_originals[-1].append((v, intensities[i]))
                i += 1
        all_fits.append((tuple(plateau_fits_originals), r_2, adj_r_2))
    else:
        for drops in itertools.product(range(len(intensities)),
                                       repeat=max_num_drops):
            drops = sorted(list(set(drops)))
            drop_ends = [d - 1 for d in drops] + [len(intensities) - 1]
            if drop_ends[0] < 0:
                drop_ends = drop_ends[1:]
            else:
                drops.insert(0, 0)
            plateau_tuples = list(zip(drops, drop_ends))
            plateaus = [intensities[start:stop + 1]
                        for start, stop in plateau_tuples]
            plateau_fits = [[np.mean(plateau)] * len(plateau)
                            for plateau in plateaus]
            merged_fit = list(itertools.chain(*plateau_fits))
            r_2 = _r_2(intensities, merged_fit)
            if r_2 < storage_r_2_cutoff:
                continue
            k = 2.0 * len(plateau_fits) - 1.0
            adj_r_2 = (1.0 -
                       (1.0 - r_2) * (len(intensities) - 1.0) /
                       (len(intensities) - k - 1.0))
            #merge original intensities into plateaus
            i = 0
            plateau_fits_originals = []
            for plateau in plateau_fits:
                plateau_fits_originals.append([])
                for v in plateau:
                    plateau_fits_originals[-1].append((v, intensities[i]))
                    i += 1
            all_fits.append((tuple(plateau_fits_originals), r_2, adj_r_2))
    return all_fits


def _cluster_fit(intensities, max_num_drops=3, zero_level=5000,
                 integer_deviation=1.4, **kwargs):
    raise NotImplementedError("This doesn't really work. Use _cluster_fit_2")
    indexed_intensities = list(enumerate(intensities))
    sorted_intensities = sorted(indexed_intensities, key=lambda x:x[1],
                                reverse=True)
    best_cluster_boundaries = None
    best_score = -1
    for partition in itertools.product(range(len(sorted_intensities)),
                                       repeat=max_num_drops):
        cluster_starts = sorted(list(set(partition) | set([0])))
        cluster_stops = ([ps - 1 for ps in cluster_starts[1:]] +
                         [len(sorted_intensities) - 1])
        cluster_boundaries = list(zip(cluster_starts, cluster_stops))
        clusters = [[i[1] for i in sorted_intensities[start:stop + 1]]
                    for start, stop in cluster_boundaries]
        cluster_means = [np.mean(cluster) for cluster in clusters]
        sorted_cluster_means = sorted(cluster_means)
        cluster_means_diff = [float(mean - sorted_cluster_means[m])
                            for m, mean in enumerate(sorted_cluster_means[1:])]
        sorted_cluster_means_diff = sorted(cluster_means_diff)
        if len(sorted_cluster_means) in [1, 2]:
            continue #TODO
        else:
            smallest_step = sorted_cluster_means_diff[0]
            if smallest_step <= zero_level:
                continue
            for divisor in (1.0, 2.0, 3.0):
                single_fluor_intensity = smallest_step / divisor
                if all(any(i / integer_deviation <=
                           diff / single_fluor_intensity
                           <= integer_deviation * i
                           for i in (1.0, 2.0, 3.0))
                       for diff in sorted_cluster_means_diff[1:]):
                    break
            else:
                continue
            fit_score = reduce(mul,
                               [2 * norm.sf(abs(intensity - cluster_means[c]),
                                            scale=zero_level)
                                for c, cluster in enumerate(clusters)
                                for intensity in cluster],
                               1.0)
            if fit_score > best_score:
                best_score = fit_score
                best_cluster_boundaries = cluster_boundaries
            elif (fit_score == best_score and
                  len(clusters) < len(best_cluster_boundaries)):
                best_score = fit_score
                best_cluster_boundaries = cluster_boundaries
    if best_cluster_boundaries is not None:
        cluster_map = {b: [i for i, v in sorted_intensities[start:stop + 1]]
                    for b, (start, stop) in enumerate(best_cluster_boundaries)}
        assert (len(set([v for s in cluster_map.itervalues() for v in s])) ==
                sum(len(set(s)) for s in cluster_map.itervalues()))
        inverse_cluster_map = {i: b
                               for b, s in cluster_map.iteritems() for i in s}
        final_fit = []
        for index, intensity in enumerate(intensities):
            if (len(final_fit) == 0 or
                inverse_cluster_map[index] != inverse_cluster_map[index - 1]):
                final_fit.append([intensity])
            else:
                final_fit[-1].append(intensity)
    else:
        final_fit = None
    return final_fit, best_score


def _cluster_fit_2(intensities, max_num_drops=3, zero_level=5000,
                   integer_deviation=1.4, scoring='gaussian',
                   largest_coincidence=3, single_fluor_min=10000,
                   gaussian_score_min=0.5, intensity_corrections=None,
                   intensity_correction_div=False, fluor_std=10000,
                   gaussian_std_max=5, min_num_drops=0, single_fluor_max=None,
                   consider_zl=True, n_init=10, zero_std=10000, **kwargs):
    if intensity_corrections is not None:
        if intensity_correction_div:
            m = float(np.amax(intensity_corrections))
            intensities = [intensity * m / intensity_corrections[i]
                           for i, intensity in enumerate(intensities)]
        else:
            intensities = [intensity - intensity_corrections[i]
                           for i, intensity in enumerate(intensities)]
    reshaped_intensities = np.array(intensities).reshape((-1, 1))
    best_clusters = None
    best_cluster_means = None
    best_score = None
    best_estimated_single_fluor_intensity = None
    for num_drops in range(min_num_drops, max_num_drops + 1):
        km = KMeans(n_clusters=num_drops + 1, init='k-means++', n_init=n_init,
                    max_iter=300, tol=0.0001, precompute_distances=True,
                    verbose=0, random_state=None, copy_x=True, n_jobs=1)
        cluster_indexes = km.fit_predict(reshaped_intensities)
        cluster_means = km.cluster_centers_
        sorted_cluster_means = sorted(cluster_means)
        if num_drops > 0:
            sorted_cluster_means_diff = \
                   sorted([float(mean - sorted_cluster_means[m])
                           for m, mean in enumerate(sorted_cluster_means[1:])])
            smallest_step = sorted_cluster_means_diff[0]
            if consider_zl:
                if smallest_step < min(zero_level, single_fluor_min):
                    continue
            elif smallest_step < single_fluor_min:
                continue
            largest_step = sorted_cluster_means_diff[-1]
            estimated_single_fluor_intensity = None
            for divisor in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)[:largest_coincidence]:
                single_fluor_intensity = smallest_step / divisor
                if single_fluor_intensity < single_fluor_min:
                    continue
                elif (single_fluor_max is not None and
                      single_fluor_intensity > single_fluor_max):
                    continue
                if all(any(i * (2.0 - integer_deviation) <=
                           diff / single_fluor_intensity
                           <= i * integer_deviation
                           for i in
                              (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)[:largest_coincidence])
                       for diff in sorted_cluster_means_diff[1:]):
                    estimated_single_fluor_intensity = single_fluor_intensity
                    break
            else:
                continue
        else:
            estimated_single_fluor_intensity = (sorted_cluster_means[0] -
                                                zero_level + zero_std)
            if estimated_single_fluor_intensity < single_fluor_min:
                continue
            elif (single_fluor_max is not None and
                  estimated_single_fluor_intensity > single_fluor_max):
                for i in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)[:largest_coincidence]:
                    new_estimate = estimated_single_fluor_intensity / i
                    if single_fluor_max >= new_estimate >= single_fluor_min:
                        estimated_single_fluor_intensity = new_estimate
                        break
                else:
                    continue
        if estimated_single_fluor_intensity < single_fluor_min:
            continue
        if (single_fluor_max is not None and
            estimated_single_fluor_intensity > single_fluor_max):
            continue
        cluster_std_coeff = \
         [max(math.sqrt(round(cluster_mean /
                              estimated_single_fluor_intensity)),
              1.0)
          if cluster_mean > zero_level and cluster_mean > 0
          else 1.0
          for c, cluster_mean in enumerate(cluster_means)]
        clusters = [[intensities[ii]
                     for ii, ci in enumerate(cluster_indexes) if ci == c]
                    for c, cluster_mean in enumerate(cluster_means)]
        if scoring == 'gaussian':
            gaussian_stds = [abs((intensity - cluster_means[c]) /
                                 (fluor_std * cluster_std_coeff[c]))
                             if cluster_means[c] > zero_level
                             else abs((intensity - cluster_means[c]) /
                                      zero_std)
                             for c, cluster in enumerate(clusters)
                             for intensity in cluster]
            if np.amax(gaussian_stds) > gaussian_std_max:
                continue
            g_scores = [norm.pdf(abs(intensity - cluster_means[c]),
                                 scale=fluor_std * cluster_std_coeff[c])
                        if cluster_means[c] > zero_level
                        else norm.pdf(abs(intensity - cluster_means[c]),
                                      scale=zero_std)
                        for c, cluster in enumerate(clusters)
                        for intensity in cluster]
            if np.amin(g_scores) < gaussian_score_min:
                continue
            fit_score = reduce(mul, g_scores, 1.0)
        elif scoring == 'std':
            raise DeprecationWarning()
            fit_score = sum([np.std(cluster)
                             for c, cluster in enumerate(clusters)])
        elif scoring == 'var':
            raise DeprecationWarning()
            fit_score = sum([np.var(cluster)
                             for c, cluster in enumerate(clusters)])
        elif scoring == 'uniform_gaussian':
            raise NotImplementedError("I have not updated this to have the "
                                      "correct scales, etc. Probably will be "
                                      "deprecated and removed in the future.")
            gaussian_stds = [abs((intensity - cluster_means[c]) / fluor_std)
                             for c, cluster in enumerate(clusters)
                             for intensity in cluster]
            if np.amax(gaussian_stds) > gaussian_std_max:
                continue
            if use_pdf:
                g_scores = [norm.pdf(abs(intensity - cluster_means[c]),
                                     scale=fluor_std)
                            for c, cluster in enumerate(clusters)
                            for intensity in cluster]
            else:
                g_scores = [2 * norm.sf(abs(intensity - cluster_means[c]),
                                        scale=fluor_std)
                            for c, cluster in enumerate(clusters)
                            for intensity in cluster]
            if np.amin(g_scores) < gaussian_score_min:
                continue
            fit_score = reduce(mul, g_scores, 1.0)
        elif scoring == 'proportional_gaussian':
            raise NotImplementedError("I have not updated this to have the "
                                      "correct scales, etc. Probably will be "
                                      "deprecated and removed in the future.")
            gaussian_stds = [abs((intensity - cluster_means[c]) /
                                 (fluor_std *
                                  max(cluster_means[c] /
                                      estimated_single_fluor_intensity, 1)))
                             for c, cluster in enumerate(clusters)
                             for intensity in cluster]
            if np.amax(gaussian_stds) > gaussian_std_max:
                continue
            if use_pdf:
                g_scores = [norm.pdf(abs(intensity - cluster_means[c]),
                                     scale=(fluor_std *
                                          max(cluster_means[c] /
                                              estimated_single_fluor_intensity,
                                              1)))
                            for c, cluster in enumerate(clusters)
                            for intensity in cluster]
            else:
                g_scores = [2 * norm.sf(abs(intensity - cluster_means[c]),
                                   scale=(fluor_std *
                                          max(cluster_means[c] /
                                              estimated_single_fluor_intensity,
                                              1)))
                            for c, cluster in enumerate(clusters)
                            for intensity in cluster]
            if np.amin(g_scores) < gaussian_score_min:
                continue
            fit_score = reduce(mul, g_scores, 1.0)
        elif scoring == 'km':
            fit_score = -1.0 * km.inertia_
        else:
            raise ValueError("not a valid scoring option")
        if best_score is None or fit_score > best_score:
            best_clusters = cluster_indexes
            best_cluster_means = cluster_means
            best_score = fit_score
            best_estimated_single_fluor_intensity = estimated_single_fluor_intensity
    if best_clusters is not None:
        final_fit = []
        for index, intensity in enumerate(intensities):
            if (len(final_fit) == 0 or
                best_clusters[index] != best_clusters[index - 1]):
                final_fit.append([intensity])
            else:
                final_fit[-1].append(intensity)
        if np.mean(final_fit[-1]) <= zero_level:
            is_zero = True
        else:
            is_zero = False
    else:
        final_fit = None
        is_zero = False
    return final_fit, best_score, is_zero, best_estimated_single_fluor_intensity


def _collate_means_into_fit(fit, reverse_order=False):
    if reverse_order:
        return tuple([[(v, np.mean(plateau))
                       for v in plateau]
                      for plateau in fit])
    else:
        return tuple([[(np.mean(plateau), v)
                       for v in plateau]
                      for plateau in fit])


def _find_experiment_levels(fits, filter_ups=False, r_2_threshold=0.7,
                            min_num_levels=None, max_num_levels=None,
                            originals_included=False,
                            use_original_values=False):
    #Computing this part takes a LONG time. Better to just get it from
    #_plateau_fit that's already done.
    #fits = [_plateau_fit(intensities=track,
    #                     max_num_drops=max_num_drops,
    #                     filter_ups=filter_ups)
    #        for track in track_intensities]
    if not originals_included:
        raw_values = np.array([v
                               for fit, r_2 in fits
                               for plateau in fit
                               for v in plateau
                               if r_2 >= r_2_threshold])
    else:
        if use_original_values:
            raw_values = np.array([v[1]
                                   for fit, r_2 in fits
                                   for plateau in fit
                                   for v in plateau
                                   if r_2 >= r_2_threshold])
        else:
            raw_values = np.array([v[0]
                                   for fit, r_2 in fits
                                   for plateau in fit
                                   for v in plateau
                                   if r_2 >= r_2_threshold])
    best_fit, best_i, best_bic = None, None, 10**10
    i_range_min = 1 if min_num_levels is None else min_num_levels
    i_range_max = len(raw_values) if max_num_levels is None else max_num_levels
    for i in range(i_range_min, i_range_max + 1):
        g = GMM(n_components=i)
        g.fit(raw_values)
        bic = g.bic(raw_values)
        if bic < best_bic:
            best_fit = g
            best_i = i
            best_bic = bic
    levels = [x for x in best_fit.means_]
    return levels, best_fit, best_bic, best_i


def _translate_plateaus_into_signal(plateaus, best_fit,
                                    originals_included=False):
    """
    Only works with downsteps.
    """
    if originals_included:
        plateaus = [[v[0] for v in p] for p in plateaus]
    for p1, p2 in _pairwise(plateaus):
        if p1[0] < p2[0]:
            raise Exception
    #sorted_levels_fits = sorted(zip(levels, best_fit),
    #                            key=lambda x:x[0],
    #                            reverse=True)
    cumulative_index = -1
    plateau_ends = []
    for plateau in plateaus:
        cumulative_index += len(plateau)
        plateau_ends.append(cumulative_index)
    plateau_starts = [0] + [e + 1 for e in plateau_ends[:-1]]
    collated_plateaus_starts_stops = zip(plateaus,
                                         plateau_starts,
                                         plateau_ends)
    #print(collated_plateaus_starts_stops)
    level_assignments = []
    for plateau, start, stop in collated_plateaus_starts_stops:
        #best_score, best_i = -10**10, None
        #for i, (level, fit) in enumerate(sorted_levels_fits):
        #    score = fit.score(plateau)[0]
        #    if score > best_score:
        #        best_score, best_i = score, i
        #assert best_i is not None
        #level_assignments.append(best_i)
        bf_index = int(best_fit.predict(plateau)[0])
        #bf_level = float(best_fit.means_[bf_index])
        level_assignments.append(bf_index)
    #print(level_assignments)
    levels = [(float(x), i) for i, x in enumerate(best_fit.means_)]
    sorted_levels = sorted(levels, key=lambda y:y[0])
    level_map = {}
    for ox, oi in levels:
        for i, (mx, mi) in enumerate(sorted_levels):
            if oi == mi:
                level_map.setdefault(oi, i)
                break
    #print(level_map)
    level_assignments = [level_map[L] for L in level_assignments]
    #print(level_assignments)
    level_drops = [L1 - L2 for L1, L2 in _pairwise(level_assignments)]
    #print(level_drops)
    signal = []
    for d, drop in enumerate(level_drops):
        drop_position = collated_plateaus_starts_stops[d][2] + 1
        signal += (('A', drop_position),) * drop
    return tuple(signal)


def _translate_plateaus_into_signal_2(plateaus, originals_included=False,
                                      adjustment=1, step_amplify=1):
    if originals_included:
        plateaus = [[v[0] for v in p] for p in plateaus]
    for p1, p2 in _pairwise(plateaus):
        if p1[0] < p2[0]:
            raise Exception
    cumulative_index = -1
    plateau_ends = []
    for plateau in plateaus[:-1]:
        cumulative_index += len(plateau)
        plateau_ends.append(cumulative_index)
    #plateau_starts = [0] + [e + 1 for e in plateau_ends[:-1]]
    #collated_plateaus_starts_stops = zip(plateaus,
    #                                     plateau_starts,
    #                                     plateau_ends)
    signal = []
    for end in plateau_ends:
        signal += (('A', end + adjustment),) * step_amplify
    return tuple(signal)

def _translate_plateaus_into_signal_3(plateaus, originals_included=False,
                                      adjustment=1, fluor_intensity=None):
    if originals_included:
        plateaus = [[v[0] for v in p] for p in plateaus]
    for p1, p2 in _pairwise(plateaus):
        if p1[0] < p2[0]:
            raise Exception
    #plateaus_scale = sorted(plateaus, reverse=True)
    plateaus_scale_f = {p[0]: int(round(float(p[0]) / fluor_intensity))
                        for p in plateaus}
    cumulative_index = -1
    plateau_ends = []
    for plateau in plateaus[:-1]:
        cumulative_index += len(plateau)
        plateau_ends.append(cumulative_index)
    #plateau_starts = [0] + [e + 1 for e in plateau_ends[:-1]]
    #collated_plateaus_starts_stops = zip(plateaus,
    #                                     plateau_starts,
    #                                     plateau_ends)
    signal = []
    for e, end in enumerate(plateau_ends):
        #step_amplify = 1
        u_p = plateaus_scale_f[plateaus[e][0]]
        L_p = plateaus_scale_f[plateaus[e + 1][0]]
        step_amplify = u_p - L_p
        signal += (('A', end + adjustment),) * step_amplify
    return tuple(signal)



def _parallel_cluster_fit(photometries, num_processes=None, channel='ch1',
                          **kwargs):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for chan, cdict in photometries.iteritems():
        if chan != channel:
            continue
        for field, fdict in cdict.iteritems():
            for (h, w), (categories, intensities, r) in fdict.iteritems():
                processes.append((pool.apply_async(_cluster_fit_2,
                                     (intensities,), kwargs),
                                     chan, field, h, w, r))
    pool.close()
    pool.join()
    fitted_photometries = {}
    collated_fits = {}
    indexed_fits = {}
    all_indexed_fits = {}
    none_fits = []
    for (p, chan, field, h, w, r) in processes:
        fit, score, is_zero, fluor_intensity = p.get()
        if fit is None:
            none_fits.append(r)
            continue
        collated_fit = _collate_means_into_fit(fit=fit)
        all_indexed_fits.setdefault(r, [chan, field, h, w, collated_fit,
                                        is_zero, fluor_intensity])
        if not _check_no_downsteps(fit):
            continue
        fitted_photometries.setdefault(chan,
            {}).setdefault(field, {}).setdefault((h, w), (fit, score, is_zero,
                                                          fluor_intensity))
        collated_fits.setdefault(chan,
              {}).setdefault(field, {}).setdefault((h, w), (collated_fit,
                                                            score, r, is_zero,
                                                            fluor_intensity))
        indexed_fits.setdefault(r, [chan, field, h, w, collated_fit,
                                    is_zero, fluor_intensity])
    signals = {}
    for chan, cdict in collated_fits.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (fit, score, r,
                         is_zero, fluor_intensity) in fdict.iteritems():
                if len(fit) == 1:
                    signal = (('A', 0),)
                else:
                    signal = _translate_plateaus_into_signal_3(plateaus=fit,
                                               originals_included=True,
                                               fluor_intensity=fluor_intensity)
                signals.setdefault((signal, is_zero), 0)
                signals[(signal, is_zero)] += 1
                indexed_fits[r] = tuple(indexed_fits[r] + [signal])
    return (fitted_photometries, collated_fits, signals, indexed_fits,
            all_indexed_fits, none_fits)


def _save_clustered_photometries_csv():
    raise NotImplementedError()


def _gmm_photometries(photometries, min_fluors=1, max_fluors=5, dpgmm=False,
                      covariance_type='full', n_init=10, n_iter=100,
                      force_num_fluors=None, cycle=None,
                      raw_photometries=None, lower_bound=None):
    if raw_photometries is None and len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    if force_num_fluors is not None:
        min_fluors = max_fluors = force_num_fluors
    if raw_photometries is None:
        raw_photometries = \
                      [intensity
                       for channel, cdict in photometries.iteritems()
                       for field, fdict in cdict.iteritems()
                       for (h, w), (category, intensities, row)
                           in fdict.iteritems()
                       for i, intensity in enumerate(intensities)
                       if cycle is None or (cycle is not None and i == cycle)]
    else:
        raw_photometries = [rp for rp in raw_photometries]
    if lower_bound is not None:
        raw_photometries = np.array([[p] for p in raw_photometries
                                     if p >= lower_bound])
    else:
        raw_photometries = np.array([[p] for p in raw_photometries])
    best_fit, best_num_fluors, best_bic = None, None, 10**10
    all_fits = []
    for num_fluors in range(min_fluors, max_fluors + 1):
        if dpgmm:
            g = DPGMM(covariance_type=covariance_type)
        else:
            g = GMM(n_components=num_fluors + 1, n_init=n_init, n_iter=n_iter,
                    covariance_type=covariance_type)
        g.fit(raw_photometries)
        bic = g.bic(raw_photometries)
        all_fits.append((g, bic))
        if bic < best_bic:
            best_fit = g
            best_num_fluors = num_fluors
            best_bic = bic
    fluor_means = [x for x in best_fit.means_]
    return (fluor_means, best_fit, best_num_fluors, best_bic, all_fits,
            raw_photometries)


def _gmm_photometries_MP(photometries, min_fluors=1, max_fluors=5, dpgmm=False,
                      covariance_type='full', num_processes=None, n_init=10,
                      n_iter=100, cycle=None, raw_photometries=None,
                      lower_bound=None):
    if raw_photometries is None and len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    best_fit, best_num_fluors, best_bic, fluor_means = None, None, 10**10, None
    all_fits = []
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), max_fluors)
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for num_fluors in range(min_fluors, max_fluors + 1):
        processes.append((pool.apply_async(_gmm_photometries,
                                           (photometries,
                                            min_fluors,
                                            max_fluors,
                                            dpgmm,
                                            covariance_type,
                                            n_init,
                                            n_iter,
                                            num_fluors,
                                            cycle,
                                            raw_photometries,
                                            lower_bound)),
                          num_fluors))
    pool.close()
    pool.join()
    for p, (process, num_fluors) in enumerate(processes):
        fm, bf, bnf, bb, af, rp = process.get()
        all_fits.append((af[0], num_fluors))
        if bb < best_bic:
            best_fit = bf
            best_num_fluors = bnf
            best_bic = bb
            fluor_means = fm
    all_fits = sorted(all_fits, key=lambda x:x[1])
    all_fits = [f for f, n in all_fits]
    fluor_means = sorted(fluor_means)
    if raw_photometries is None:
        raw_photometries = \
             np.array([intensity
                       for channel, cdict in photometries.iteritems()
                       for field, fdict in cdict.iteritems()
                       for (h, w), (category, intensities, row)
                           in fdict.iteritems()
                       for i, intensity in enumerate(intensities)
                       if cycle is None or (cycle is not None and i == cycle)])
    return (fluor_means, best_fit, best_num_fluors, best_bic, all_fits,
            raw_photometries)


def _per_cycle_gmm_MP(photometries, min_fluors=1, max_fluors=5, dpgmm=False,
                      covariance_type='full', num_processes=None, n_init=10,
                      n_iter=100, cycles=None, lower_bound=None):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    if cycles is None:
        cdict = next(photometries.itervalues())
        fdict = next(cdict.itervalues())
        category, intensities, row = next(fdict.itervalues())
        cycles = tuple([i for i, intensity in enumerate(intensities)])
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(),
                            max_fluors * len(cycles))
    raw_photometries = \
       {cycle:
        np.array([intensity
                  for channel, cdict in photometries.iteritems()
                  for field, fdict in cdict.iteritems()
                  for (h, w), (category, intensities, row) in fdict.iteritems()
                  for i, intensity in enumerate(intensities)
                  if i == cycle])
        for cycle in cycles}
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for cycle in cycles:
        for num_fluors in range(min_fluors, max_fluors + 1):
            process = pool.apply_async(_gmm_photometries,
                                       (photometries,
                                        min_fluors,
                                        max_fluors,
                                        dpgmm,
                                        covariance_type,
                                        n_init,
                                        n_iter,
                                        num_fluors,
                                        cycle,
                                        None,
                                        lower_bound))
            processes.append((process, num_fluors, cycle))
    pool.close()
    pool.join()
    all_fits = {cycle: [] for cycle in cycles}
    #each entry in all_fit_scores =
    #0         1                2         3
    #best_fit, best_num_fluors, best_bic, fluor_means
    all_fit_scores = {cycle: [None, None, 10**10, None]
                      for cycle in cycles}
    for p, (process, num_fluors, cycle) in enumerate(processes):
        fm, bf, bnf, bb, af, rp = process.get()
        all_fits[cycle].append((af[0], num_fluors))
        best_fit, best_num_fluors, best_bic, fluor_means = \
                                                          all_fit_scores[cycle]
        if bb < best_bic:
            best_fit = bf
            best_num_fluors = bnf
            best_bic = bb
            fluor_means = fm
            all_fit_scores[cycle] = [best_fit, best_num_fluors, best_bic,
                                     fluor_means]
    for cycle, fits in all_fits.items():
        all_fits[cycle] = tuple(sorted(fits, key=lambda x:x[1]))
    for cycle, fits in all_fits.items():
        all_fits[cycle] = tuple([f for f, n in fits])
    for cycle, (best_fit, best_num_fluors,
                best_bic, fluor_means) in all_fit_scores.items():
        all_fit_scores[cycle] = (best_fit, best_num_fluors, best_bic,
                                 tuple(sorted(fluor_means)))
    return all_fit_scores, all_fits, raw_photometries


def _gmm_adjust(photometries, mu_zero, sigma_zero, mu_one, sigma_one,
                per_cycle_m0s0m1s1):
    per_cycle_coefficients = {cycle: float(mu_one - mu_zero) / (cm1 - cm0)
                              for cycle, (cm0, cs0, cm1, cs1)
                              in per_cycle_m0s0m1s1.iteritems()}
    per_cycle_photometries = {}
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                corrected_intensities = \
                              [per_cycle_coefficients[i] *
                               (intensity - per_cycle_m0s0m1s1[i][0]) + mu_zero
                               for i, intensity in enumerate(intensities)]
                per_cycle_photometries.setdefault(
                             channel, {}).setdefault(
                               field, {}).setdefault(
                                (h, w), (category, corrected_intensities, row))
    return per_cycle_photometries, per_cycle_coefficients


def _remainder_adjust(photometries, num_frames, minimum_r_per_field=5):
    remainder_values = {}
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                if set(category) != set([True]):
                    continue
                remainder_values.setdefault(channel, {}).setdefault(field,
                                               [[] for f in range(num_frames)])
                for frame, intensity in enumerate(intensities):
                    remainder_values[channel][field][frame].append(intensity)
    remainder_adjustments = {}
    for channel, cdict in remainder_values.iteritems():
        for field, remainder_lists in cdict.iteritems():
            if len(remainder_lists[0]) < minimum_r_per_field:
                continue
            remainder_medians = [np.median(remainder_list)
                                 for remainder_list in remainder_lists]
            adjustments = [median - remainder_medians[0]
                           for median in remainder_medians]
            remainder_adjustments.setdefault(channel, {}).setdefault(field,
                                                                   adjustments)
    adjusted_photometries = {}
    for channel, cdict in remainder_adjustments.iteritems():
        adjusted_photometries.setdefault(channel, {})
        for field, adjustments in cdict.iteritems():
            adjusted_photometries[channel].setdefault(field, {})
            fdict = photometries[channel][field]
            for (h, w), (category, intensities, row) in fdict.iteritems():
                adjusted_intensities = [intensity - adjustments[frame]
                                for frame, intensity in enumerate(intensities)]
                adjusted_photometries[channel][field].setdefault((h, w),
                                         (category, adjusted_intensities, row))
    return adjusted_photometries, remainder_adjustments


def _remainder_adjust_2(photometries, num_frames, minimum_r_per_field=5):
    adjustment_ratios = {}
    for channel, cdict in photometries.iteritems():
        adjustment_ratios.setdefault(channel, {})
        for field, fdict in cdict.iteritems():
            adjustment_ratios[channel].setdefault(field,
                                               [[] for n in range(num_frames)])
            for (h, w), (category, intensities, row) in fdict.iteritems():
                if set(category) == set([True]):
                    m = np.median(intensities)
                    for i, intensity in enumerate(intensities):
                        r = float(intensity - m) / m
                        adjustment_ratios[channel][field][i].append(r)
    adjustment_ratio_medians = {}
    for channel, cdict in adjustment_ratios.iteritems():
        for field, field_ratios in cdict.iteritems():
            if any([len(ratios) < minimum_r_per_field
                    for ratios in field_ratios]):
                continue
            adjustment_ratio_medians.setdefault(channel, {}).setdefault(field,
                                [np.median(ratios) for ratios in field_ratios])
    adjusted_photometries = {}
    for channel, cdict in photometries.iteritems():
        if channel not in adjustment_ratio_medians:
            continue
        else:
            adjusted_photometries.setdefault(channel, {})
        for field, fdict in cdict.iteritems():
            if field not in adjustment_ratio_medians[channel]:
                continue
            else:
                adjusted_photometries[channel].setdefault(field, {})
            ar = adjustment_ratio_medians[channel][field]
            for (h, w), (category, intensities, row) in fdict.iteritems():
                adjusted_intensities = [intensity * (1.0 - ar[i])
                                    for i, intensity in enumerate(intensities)]
                adjusted_photometries[channel][field].setdefault((h, w),
                                         (category, adjusted_intensities, row))
    return adjusted_photometries, adjustment_ratio_medians


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def _parameter_sweep(photometries_file,
                     clustering_parameters=None,
                     zero_fluor_std_amplifier=1.0,
                     one_fluor_std_amplifier=1.0,
                     fname_hash=None,
                     head_truncate=0, tail_truncate=0,
                     downstep_filtered=True, adjust_photometries=True,
                     minimum_r_per_field=5, max_fluors=2,
                     covariance_type='full', n_init=10,
                     n_iter=100, channel='ch1',
                     clustering_parameters_A_delta=None,
                     clustering_parameters_M_delta=None):
    raise NotImplementedError("Deprecated. Use _parameter_sweep_2")
    print("A")
    photometries, row_photometries = \
               read_track_photometries_csv(photometries_file,
                                           head_truncate=head_truncate,
                                           tail_truncate=tail_truncate,
                                           downstep_filtered=downstep_filtered)
    num_frames = len(next(row_photometries.iteritems())[1][4])
    if adjust_photometries:
        adjusted_photometries, remainder_adjustments = \
               _remainder_adjust(photometries, num_frames, minimum_r_per_field)
        use_photometries = adjusted_photometries
    else:
        use_photometries = photometries
        remainder_adjustments = None
    print(" B")
    (fluor_means, best_fit, best_num_fluors, best_bic, all_fits,
     raw_photometries) = \
           _gmm_photometries_MP(use_photometries, max_fluors=max_fluors,
                                covariance_type=covariance_type, n_init=n_init,
                                n_iter=n_iter)
    dual_gmm = all_fits[1][0]
    dual_gmm_means = [float(m) for m in dual_gmm.means_]
    dual_gmm_weights = [float(w) for w in dual_gmm.weights_]
    dual_gmm_vars = [float(v) for v in dual_gmm.covars_]
    dual_gmm_stds = [math.sqrt(v) for v in dual_gmm_vars]
    dual_gmm_stats = sorted(zip(dual_gmm_means, dual_gmm_weights,
                                dual_gmm_vars, dual_gmm_stds),
                            key=lambda x:x[0])
    zero_fluor_mean = dual_gmm_stats[0][0]
    zero_fluor_std = dual_gmm_stats[0][3] * zero_fluor_std_amplifier
    one_fluor_mean = dual_gmm_stats[1][0]
    one_fluor_std = dual_gmm_stats[1][3] * one_fluor_std_amplifier
    print("  C")
    default_clustering_parameters = \
              {'max_num_drops': 5,
               'zero_level': zero_fluor_mean + zero_fluor_std,
               'integer_deviation': 1.4,
               'scoring': 'gaussian',
               #'scoring': 'proportional_gaussian',
               #'scoring': 'uniform_gaussian',
               'gaussian_score_min': 0.0,
               'gaussian_std_max': 3,
               'largest_coincidence': 5,
               'single_fluor_min': one_fluor_mean - one_fluor_std,
               'single_fluor_max': one_fluor_mean + one_fluor_std,
                #'intensity_corrections': [22333.0, 27287.0, 26819.0, 27125.0,
                #                          25724.0, 26157.0, 25801.0, 27517.0,
                #                          28746.0, 28722.0, 28790.0, 30392.0],
               'intensity_correction_div': True,
               'use_pdf': True,
               'algorithm': '_cluster_fit_2',
               'fluor_std': one_fluor_std,
               'channel': channel,
               'version': '2016mar21_04:36'}
    if clustering_parameters is not None:
        default_clustering_parameters.update(clustering_parameters)
    if clustering_parameters_A_delta is not None:
        for k, v in clustering_parameters_A_delta.iteritems():
            default_clustering_parameters[k] += v
    if clustering_parameters_M_delta is not None:
        for k, v in clustering_parameters_M_delta.iteritems():
            default_clustering_parameters[k] *= v
    print("   D")
    results = \
        (fitted_photometries, collated_fits, signals, indexed_fits,
         all_indexed_fits, none_fits) = \
                         _parallel_cluster_fit(use_photometries,
                                               **default_clustering_parameters)
    print("    E")
    if fname_hash is None:
        timestamp_epoch = int(round(time.time()))
        #fname_hash = pflib._epoch_to_hash(timestamp_epoch)
        fname_hash = str(timestamp_epoch)
    save_parameters = \
                    (photometries_file, head_truncate, tail_truncate,
                     downstep_filtered, adjust_photometries,
                     minimum_r_per_field, max_fluors, covariance_type, n_init,
                     n_iter, channel, default_clustering_parameters)
    save_gmm = (zero_fluor_mean, zero_fluor_std, one_fluor_mean, one_fluor_std,
                best_fit)
    save_modifiers = (zero_fluor_std_amplifier, one_fluor_std_amplifier,
                      clustering_parameters['integer_deviation'])
    cPickle.dump((results, save_parameters, save_gmm, remainder_adjustments,
                  save_modifiers),
                 open(basename(photometries_file) +
                      fname_hash + '_results.pkl', 'w'))
    print("     F")
    return results, save_parameters


def _parameter_sweep_2(photometries_file,
                       clustering_parameters=None,
                       zero_fluor_std_amplifier=1.0,
                       one_fluor_std_amplifier=1.0,
                       fname_hash=None,
                       head_truncate=0, tail_truncate=0,
                       downstep_filtered=True, adjust_photometries=False,
                       minimum_r_per_field=5, max_fluors=10,
                       covariance_type='full', n_init=10,
                       n_iter=100, channel='ch1',
                       clustering_parameters_A_delta=None,
                       clustering_parameters_M_delta=None):
    print("A")
    photometries, row_photometries = \
               read_track_photometries_csv(photometries_file,
                                           head_truncate=head_truncate,
                                           tail_truncate=tail_truncate,
                                           downstep_filtered=downstep_filtered)
    num_frames = len(next(row_photometries.iteritems())[1][4])
    if adjust_photometries:
        adjusted_photometries, remainder_adjustments = \
               _remainder_adjust(photometries, num_frames, minimum_r_per_field)
        use_photometries = adjusted_photometries
    else:
        use_photometries = photometries
        remainder_adjustments = None
    print(" B")
    (fluor_means, best_fit, best_num_fluors, best_bic, all_fits,
     raw_photometries) = \
           _gmm_photometries_MP(use_photometries, max_fluors=max_fluors,
                                covariance_type=covariance_type, n_init=n_init,
                                n_iter=n_iter)
    #dual_gmm = all_fits[1][0]
    #dual_gmm_means = [float(m) for m in dual_gmm.means_]
    #dual_gmm_weights = [float(w) for w in dual_gmm.weights_]
    #dual_gmm_vars = [float(v) for v in dual_gmm.covars_]
    #dual_gmm_stds = [math.sqrt(v) for v in dual_gmm_vars]
    #dual_gmm_stats = sorted(zip(dual_gmm_means, dual_gmm_weights,
    #                            dual_gmm_vars, dual_gmm_stds),
    #                        key=lambda x:x[0])
    best_fit_means = [float(m) for m in best_fit.means_]
    best_fit_weights = [float(w) for w in best_fit.weights_]
    best_fit_vars = [float(v) for v in best_fit.covars_]
    best_fit_stds = [math.sqrt(v) for v in best_fit_vars]
    best_fit_stats = sorted(zip(best_fit_means, best_fit_weights,
                                best_fit_vars, best_fit_stds),
                            key=lambda x:x[1], reverse=True)
    zero_fluor_mean = best_fit_stats[0][0]
    zero_fluor_std = best_fit_stats[0][3] * zero_fluor_std_amplifier
    one_fluor_mean = best_fit_stats[1][0]
    one_fluor_std = best_fit_stats[1][3] * one_fluor_std_amplifier
    print("  C")
    default_clustering_parameters = \
              {'max_num_drops': 5,
               'zero_level': zero_fluor_mean + zero_fluor_std,
               'integer_deviation': 1.4,
               'scoring': 'gaussian',
               #'scoring': 'proportional_gaussian',
               #'scoring': 'uniform_gaussian',
               'gaussian_score_min': 0.0,
               'gaussian_std_max': 3,
               'largest_coincidence': 5,
               'single_fluor_min': one_fluor_mean - one_fluor_std,
               'single_fluor_max': one_fluor_mean + one_fluor_std,
                #'intensity_corrections': [22333.0, 27287.0, 26819.0, 27125.0,
                #                          25724.0, 26157.0, 25801.0, 27517.0,
                #                          28746.0, 28722.0, 28790.0, 30392.0],
               'intensity_correction_div': True,
               'use_pdf': True,
               'algorithm': '_cluster_fit_2',
               'fluor_std': one_fluor_std,
               'channel': channel,
               'version': '2016mar21_04:36'}
    if clustering_parameters is not None:
        default_clustering_parameters.update(clustering_parameters)
    if clustering_parameters_A_delta is not None:
        for k, v in clustering_parameters_A_delta.iteritems():
            default_clustering_parameters[k] += v
    if clustering_parameters_M_delta is not None:
        for k, v in clustering_parameters_M_delta.iteritems():
            default_clustering_parameters[k] *= v
    print("   D")
    results = \
        (fitted_photometries, collated_fits, signals, indexed_fits,
         all_indexed_fits, none_fits) = \
                         _parallel_cluster_fit(use_photometries,
                                               **default_clustering_parameters)
    print("    E")
    if fname_hash is None:
        timestamp_epoch = int(round(time.time()))
        #fname_hash = pflib._epoch_to_hash(timestamp_epoch)
        fname_hash = str(timestamp_epoch)
    save_parameters = \
                    (photometries_file, head_truncate, tail_truncate,
                     downstep_filtered, adjust_photometries,
                     minimum_r_per_field, max_fluors, covariance_type, n_init,
                     n_iter, channel, default_clustering_parameters)
    save_gmm = (zero_fluor_mean, zero_fluor_std, one_fluor_mean, one_fluor_std,
                best_fit, best_fit_stats)
    save_modifiers = (zero_fluor_std_amplifier, one_fluor_std_amplifier,
                      clustering_parameters['integer_deviation'])
    cPickle.dump((results, save_parameters, save_gmm, remainder_adjustments,
                  save_modifiers),
                 open(basename(photometries_file) +
                      fname_hash + '_results.pkl', 'w'))
    print("     F")
    return results, save_parameters




def _parallel_parameter_sweep(photometries_filepath, pdict=None,
                              num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    integer_deviation_range = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    zero_fluor_std_amplifier = [0.50, 0.60, 0.70, 0.80, 0.90, 1.0,
                                1.10, 1.20, 1.30, 1.40, 1.50, 1.60,
                                1.70, 1.80, 1.90, 2.00, 2.10, 2.20]
    one_fluor_std_amplifier = [0.50, 0.60, 0.70, 0.80, 0.90, 1.0,
                               1.10, 1.20, 1.30, 1.40, 1.50, 1.60,
                               1.70, 1.80, 1.90, 2.00, 2.10, 2.20]
    #pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    pool = MyPool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for int_d, zfsa, ofsa in itertools.product(integer_deviation_range,
                                               zero_fluor_std_amplifier,
                                               one_fluor_std_amplifier):
        raise Exception("Note to self: time.time() is not high resolution "
                        "enough to differentiate items in this loop.")
        timestamp_epoch = int(round(time.time()))
        fname_hash = str(timestamp_epoch)
        clustering_parameters = {'integer_deviation': int_d}
        processes.append(pool.apply_async(_parameter_sweep,
                                          (photometries_filepath,
                                           clustering_parameters,
                                           zfsa, ofsa, fname_hash)))
    pool.close()
    pool.join()

def _ps_results_analysis():
    raise NotImplementedError()

def _intensities_to_signal_lognormal(intensities, mu_zero=0, sigma_zero=20000,
                                     mu_one=60000, max_possible=5,
                                     allow_multidrop=False):
    intensities = [intensity - mu_zero for intensity in intensities]
    zero_fluor = mu_zero + 2.0 * sigma_zero
    one_fluor = mu_one - mu_zero
    two_fluor = 2.0 * one_fluor
    log_one_fluor, log_two_fluor = log(one_fluor), log(two_fluor)
    half_log_fluor = np.mean((log_one_fluor, log_two_fluor)) - log_one_fluor
    #log_fluor_boundaries[1] = upper_log_one
    #log_fluor_boundaries[2] = upper_log_two
    #log_fluor_boundaries[3] = upper_log_three
    #...etc
    #log_fluor_boundaries = [log_one_fluor + (2 * i + 1) * half_log_fluor
    #                        for i in range(max_possible)]
    log_fluor_boundaries = [np.mean([log(one_fluor + i * one_fluor),
                                     log(one_fluor + (i + 1) * one_fluor)])
                            for i in range(max_possible + 1)]
    log_fluor_means = [log(one_fluor + i * one_fluor)
                       for i in range(max_possible + 2)]
    log_max_intensity = log(max(max(intensities), 1))
    lmii = max_possible
    for i, lfb in enumerate(log_fluor_boundaries):
        if log_max_intensity > lfb:
            continue
        else:
            lmii = i + 2
            break
    best_seq = None
    best_score = -1
    log_intensities = [log(intensity) if intensity > zero_fluor else -100
                       for intensity in intensities]
    best_log_score = None
    best_intensity_scores = None
    for seq in combinations_with_replacement(reversed(range(lmii + 1)),
                                             len(intensities)):
        if not allow_multidrop:
            seq_diff = [seq[i] - s for i, s in enumerate(seq[1:])]
            if max(seq_diff) > 1:
                continue
        if any([(intensity <= zero_fluor and seq[i] != 0) or
                (intensity > zero_fluor and seq[i] == 0)
                for i, intensity in enumerate(intensities)]):
            continue
        intensity_scores = [norm.pdf(log_intensity,
                                     loc=log_fluor_means[seq[i] - 1],
                                     scale=half_log_fluor)
                            for i, log_intensity in enumerate(log_intensities)
                            if log_intensity > 0]
        log_intensity_scores = \
                            [norm.logpdf(log_intensity,
                                         loc=log_fluor_means[seq[i] - 1],
                                         scale=half_log_fluor)
                             for i, log_intensity in enumerate(log_intensities)
                             if log_intensity > 0]
        total_score = reduce(mul, intensity_scores, 1.0)
        total_log_score = sum([log_score
                               for log_score in log_intensity_scores])
        if total_score > best_score:
            best_seq = seq
            best_score = total_score
            best_log_score = total_log_score
            best_intensity_scores = intensity_scores
    if best_seq is not None:
        signal_TFn = [best_seq[f] - fc for f, fc in enumerate(best_seq[1:])]
        signal = []
        for i, tf in enumerate(signal_TFn):
            if tf > 0:
                signal += [('A', i + 1)] * tf
            elif tf < 0:
                raise Exception()
        signal = tuple(signal)
        if len(signal) == 0:
            signal = [('A', 0)]
        signal = tuple(signal)
        if best_seq[-1] == 0:
            is_zero = True
        else:
            is_zero = False
    else:
        signal = None
        is_zero = None
    return (signal, is_zero, best_seq, lmii, best_score, best_log_score,
            best_intensity_scores)


def _per_cycle_intensities_to_signal_lognormal(intensities,
                                               per_cycle_parameters,
                                               max_possible=5):
    """
    per_cycle_parameters = [(mu_zero, sigma_zero, mu_one), #cycle 0
                            (mu_zero, sigma_zero, mu_one), #cycle 1
                            ... etc]
    """
    raise NotImplementedError()

def _photometries_lognormal_fit_MP(photometries, mu_zero=0, sigma_zero=20000,
                                   mu_one=60000, max_possible=5,
                                   num_processes=None,
                                   per_cycle_parameters=None,
                                   allow_multidrop=False):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                if per_cycle_parameters is None:
                    process = pool.apply_async(
                                              _intensities_to_signal_lognormal,
                                               (intensities,
                                                mu_zero,
                                                sigma_zero,
                                                mu_one,
                                                max_possible,
                                                allow_multidrop))
                else:
                    process = pool.apply_async(
                                    _per_cycle_intensities_to_signal_lognormal,
                                               (intensities,
                                                mu_zero,
                                                sigma_zero,
                                                mu_one,
                                                max_possible,
                                                per_cycle_parameters))
                processes.append((process, channel, field, h, w, row, category,
                                  intensities))
    pool.close()
    pool.join()
    signals = {}
    none_count = 0
    total_count = 0
    all_fit_info = []
    for i, (process, channel, field, h, w, row, category,
            intensities) in enumerate(processes):
        total_count += 1
        (signal, is_zero, best_seq, lmii, best_score, best_log_score,
         best_intensity_scores) = process.get()
        all_fit_info.append((channel, field, h, w, row, category, intensities,
                             signal, is_zero, best_seq, lmii, best_score,
                             best_log_score, best_intensity_scores))
        if signal is None:
            none_count += 1
        else:
            signals.setdefault((signal, is_zero), 0)
            signals[(signal, is_zero)] += 1
    return signals, total_count, none_count, all_fit_info


def optimal_bin_size(raw_photometries, bin_array=None):
    """
    Shimazaki & Shinomoto; 10.1162/neco.2007.19.6.1503
    http://toyoizumilab.brain.riken.jp/hideaki/res/histogram.html#Python1D
    """
    raw_photometries_min = min(raw_photometries)
    raw_photometries_max = max(raw_photometries)
    if bin_array is None:
        min_n_bins, max_n_bins = 10, 100
        bin_array = np.array(range(min_n_bins, max_n_bins + 1))
    bin_size_vector = (float(raw_photometries_max - raw_photometries_min) /
                       bin_array)
    cost_array = np.zeros(shape=(bin_size_vector.size, 1))
    for i, bin_size in enumerate(bin_size_vector):
        edges = np.linspace(raw_photometries_min,
                            raw_photometries_max,
                            bin_array[i] + 1)
        hist, bins = np.histogram(a=raw_photometries, bins=edges)
        cost_array[i] = ((2.0 * np.mean(hist) - np.var(hist, ddof=0)) /
                         bin_size**2)
    min_cost = np.amin(cost_array)
    return min_cost, np.where(cost_array == min_cost), cost_array


def optimal_bin_size_MP(raw_photometries, num_processes=None, min_n_bins=10,
                        max_n_bins=1000):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    bin_array = np.array(range(min_n_bins, max_n_bins + 1))
    split_bin_array = np.array_split(bin_array,
                                     min(num_processes, bin_array.size))
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for a, array in enumerate(split_bin_array):
        process = pool.apply_async(optimal_bin_size, (raw_photometries, array))
        processes.append((process, a, array))
    pool.close()
    pool.join()
    results = []
    concatenated_cost_array = [[] for a in split_bin_array]
    for i, (process, a, array) in enumerate(processes):
        min_cost, obs, cost_array = process.get()
        results.append((min_cost,
                        int(obs[0]) + min_n_bins +
                            sum([len(array) for array in split_bin_array[:a]]),
                        cost_array, a, array))
        concatenated_cost_array[a] = cost_array
    raw_cost_array = [A.copy() for A in concatenated_cost_array]
    concatenated_cost_array = \
                        np.concatenate(np.concatenate(concatenated_cost_array))
    min_result = min(results, key=lambda x:x[0])
    return min_result, results, concatenated_cost_array, raw_cost_array


def _get_m0Dm1(raw_photometries, optimal_bin_number=None):
    """Returns m0, m1, and dividing line between normal f0 and lognormal f1+"""
    if optimal_bin_number is None:
        min_result, results, concatenated_cost_array, raw_cost_array = \
            optimal_bin_size_MP(raw_photometries=raw_photometries,
                                num_processes=None,
                                min_n_bins=10, max_n_bins=10000)
        optimal_bin_number = min_result[1]
    hist, bins = np.histogram(a=raw_photometries, bins=optimal_bin_number)
    depth_array = np.zeros_like(hist)
    #global maximum (m0) = alpha
    #second_highest (m1) = beta
    #valley (D) = gamma
    for (gamma_index,), gamma_value in np.ndenumerate(hist):
        if gamma_index == 0 or gamma_index == hist.shape[0] - 1:
            continue
        L_hist, R_hist = hist[:gamma_index], hist[gamma_index + 1:]
        L_max, R_max = np.amax(L_hist), np.amax(R_hist)
        if gamma_value > L_max or gamma_value > R_max:
            continue
        depth = min(L_max, R_max) - gamma_value
        depth_array[gamma_index] = depth
    gamma_index, gamma = np.argmax(depth_array), np.amax(depth_array)
    alpha_index = np.argmax(hist[:gamma_index])
    alpha = np.amax(hist[:gamma_index])
    beta_index = gamma_index + 1 + np.argmax(hist[gamma_index + 1:])
    beta = np.amax(hist[gamma_index + 1:])
    raw_photometries_min = min(raw_photometries)
    raw_photometries_max = max(raw_photometries)
    mapping_factor = (float(raw_photometries_max - raw_photometries_min) /
                      optimal_bin_number)
    def map_bin_to_photometry(bi):
        return raw_photometries_min + mapping_factor * bi
    return (optimal_bin_number,
            alpha, alpha_index, beta, beta_index, gamma, gamma_index,
            map_bin_to_photometry(alpha_index),
            map_bin_to_photometry(beta_index),
            map_bin_to_photometry(gamma_index))


def _intensities_to_signal_lognormal_v2(intensities,
                                        alpha, beta, gamma,
                                        max_possible=5, allow_multidrop=False,
                                        allow_upsteps=False,
                                        upstep_rapid_classify=True):
    #Baseline shift everything based on alpha
    beta -= alpha
    gamma -= alpha
    intensities = [intensity - alpha for intensity in intensities]
    if allow_upsteps and upstep_rapid_classify:
        zeros = [False if intensity < gamma else True
                 for intensity in intensities]
        if not (sorted(zeros, reverse=True) == zeros and zeros[0]):
            return (None, None, None, None, None, None, None)
    #zero_fluor = mu_zero + 2.0 * sigma_zero #This is now gamma
    #one_fluor = mu_one - mu_zero #This is now beta
    two_fluor = 2.0 * beta
    log_one_fluor, log_two_fluor = log(beta), log(two_fluor)
    half_log_fluor = np.mean((log_one_fluor, log_two_fluor)) - log_one_fluor
    #log_fluor_boundaries[1] = upper_log_one
    #log_fluor_boundaries[2] = upper_log_two
    #log_fluor_boundaries[3] = upper_log_three
    #...etc
    #log_fluor_boundaries = [log_one_fluor + (2 * i + 1) * half_log_fluor
    #                        for i in range(max_possible)]
    log_fluor_boundaries = [np.mean([log(beta + i * beta),
                                     log(beta + (i + 1) * beta)])
                            for i in range(max_possible + 1)]
    log_fluor_means = [log(beta + i * beta)
                       for i in range(max_possible + 2)]
    log_max_intensity = log(max(max(intensities), 1))
    lmii = max_possible
    for i, lfb in enumerate(log_fluor_boundaries):
        if log_max_intensity > lfb:
            continue
        else:
            lmii = i + 2
            break
    best_seq = None
    best_score = -1
    log_intensities = [log(intensity) if intensity > gamma else -100
                       for intensity in intensities]
    best_log_score = None
    best_intensity_scores = None
    if allow_upsteps:
        if upstep_rapid_classify:
            zeros_count = len([z for z in zeros if not z])
            X = ([range(1, lmii + 1)] * (len(intensities) - zeros_count) +
                 [[0]] * zeros_count)
            iterator = product(*X, repeat=1)
        else:
            iterator = product(reversed(range(lmii + 1)),
                               repeat=len(intensities))
    else:
        iterator = combinations_with_replacement(reversed(range(lmii + 1)),
                                                 len(intensities))
    for seq in iterator:
        if not allow_multidrop:
            seq_diff = [seq[i] - s for i, s in enumerate(seq[1:])]
            if max(seq_diff) > 1:
                continue
        if any([(intensity <= gamma and seq[i] != 0) or
                (intensity > gamma and seq[i] == 0)
                for i, intensity in enumerate(intensities)]):
            continue
        intensity_scores = [norm.pdf(log_intensity,
                                     loc=log_fluor_means[seq[i] - 1],
                                     scale=half_log_fluor)
                            for i, log_intensity in enumerate(log_intensities)
                            if log_intensity > 0]
        log_intensity_scores = \
                            [norm.logpdf(log_intensity,
                                         loc=log_fluor_means[seq[i] - 1],
                                         scale=half_log_fluor)
                             for i, log_intensity in enumerate(log_intensities)
                             if log_intensity > 0]
        total_score = reduce(mul, intensity_scores, 1.0)
        total_log_score = sum([log_score
                               for log_score in log_intensity_scores])
        if total_score > best_score:
            best_seq = seq
            best_score = total_score
            best_log_score = total_log_score
            best_intensity_scores = intensity_scores
    if best_seq is not None:
        signal_TFn = [best_seq[f] - fc for f, fc in enumerate(best_seq[1:])]
        signal = []
        for i, tf in enumerate(signal_TFn):
            if tf > 0:
                signal += [('A', i + 1)] * tf
            elif tf < 0:
                signal = None
                break
        if signal is not None:
            signal = tuple(signal)
            if len(signal) == 0:
                signal = [('A', 0)]
            signal = tuple(signal)
            if best_seq[-1] == 0:
                is_zero = True
            else:
                is_zero = False
        else:
            is_zero = None
    else:
        signal = None
        is_zero = None
    return (signal, is_zero, best_seq, lmii, best_score, best_log_score,
            best_intensity_scores)


def _photometries_lognormal_fit_MP_v2(photometries, alpha, beta, gamma,
                                      max_possible=5,
                                      num_processes=None, allow_upsteps=False,
                                      allow_multidrop=False,
                                      upstep_rapid_classify=True):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                process = pool.apply_async(
                                           _intensities_to_signal_lognormal_v2,
                                               (intensities,
                                                alpha,
                                                beta,
                                                gamma,
                                                max_possible,
                                                allow_multidrop,
                                                allow_upsteps,
                                                upstep_rapid_classify))
                processes.append((process, channel, field, h, w, row, category,
                                  intensities))
    pool.close()
    pool.join()
    signals = {}
    none_count = 0
    total_count = 0
    all_fit_info = []
    for i, (process, channel, field, h, w, row, category,
            intensities) in enumerate(processes):
        total_count += 1
        (signal, is_zero, best_seq, lmii, best_score, best_log_score,
         best_intensity_scores) = process.get()
        all_fit_info.append((channel, field, h, w, row, category, intensities,
                             signal, is_zero, best_seq, lmii, best_score,
                             best_log_score, best_intensity_scores))
        if signal is None:
            none_count += 1
        else:
            signals.setdefault((signal, is_zero), 0)
            signals[(signal, is_zero)] += 1
    return signals, total_count, none_count, all_fit_info


def _lognormal_nearest_neighbor(intensities, alpha, beta, gamma,
                                max_possible=20):
    beta -= alpha
    gamma -= alpha
    intensities = [intensity - alpha for intensity in intensities]
    log_fluor_means = [log(beta + i * beta) for i in range(max_possible + 2)]
    nearest_neighbors = []
    for intensity in intensities:
        if intensity < gamma:
            nearest_neighbors.append(0)
        else:
            log_intensity = log(intensity)
            distances = [abs(log_intensity - mean) for mean in log_fluor_means]
            nearest = np.argmin(distances) + 1
            nearest_neighbors.append(nearest)
    signal_TFn = [nearest_neighbors[f] - fc
                  for f, fc in enumerate(nearest_neighbors[1:])]
    signal = []
    for i, tf in enumerate(signal_TFn):
        if tf > 0:
            signal += [('A', i + 1)] * tf
        elif tf < 0:
            signal = None
            break
    if signal is not None:
        signal = tuple(signal)
        if len(signal) == 0:
            signal = [('A', 0)]
        signal = tuple(signal)
        if nearest_neighbors[-1] == 0:
            is_zero = True
        else:
            is_zero = False
    else:
        is_zero = None
    return signal, is_zero, nearest_neighbors


def _lognormal_nearest_neighbor_MP(photometries, alpha, beta, gamma,
                                   max_possible=20, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                process = pool.apply_async(_lognormal_nearest_neighbor,
                                           (intensities,
                                            alpha,
                                            beta,
                                            gamma,
                                            max_possible))
                processes.append((process, channel, field, h, w, row, category,
                                  intensities))
    pool.close()
    pool.join()
    all_fit_info = []
    signals = {}
    for i, (process, channel, field, h, w, row, category,
            intensities) in enumerate(processes):
        signal, is_zero, nearest_neighbors = process.get()
        all_fit_info.append((channel, field, h, w, row, category, intensities,
                             signal, is_zero, nearest_neighbors, None, None,
                             None, None))
        if signal is not None:
            signals.setdefault((signal, is_zero), 0)
            signals[(signal, is_zero)] += 1
    return signals, all_fit_info


def fwhm_method(raw_photometries, optimal_bin_number=None):
    (optimal_bin_number,
     alpha, alpha_index,
     beta, beta_index,
     gamma, gamma_index,
     alpha_photometry,
     beta_photometry,
     gamma_photometry) = \
                     _get_m0Dm1(raw_photometries=raw_photometries,
                                optimal_bin_number=optimal_bin_number)
    #compute alpha_sigma
    sub_alpha_photometries = [photometry for photometry in raw_photometries
                              if photometry <= alpha_photometry]
    min_result, results, concatenated_cost_array, raw_cost_array = \
                   optimal_bin_size_MP(raw_photometries=sub_alpha_photometries)
    SAP_optimal_bin_number = min_result[1]
    SAP_hist, SAP_bins = np.histogram(a=sub_alpha_photometries,
                                      bins=SAP_optimal_bin_number)
    SAP_hwhm = (gamma_photometry - alpha_photometry) / 2.0
    for (i,), h in sorted(np.ndenumerate(SAP_hist), key=lambda x:x[0][0]):
        if h < alpha / 2.0:
            continue
        else:
            sub_alpha_photometries_min = np.amin(sub_alpha_photometries)
            sub_alpha_photometries_max = np.amax(sub_alpha_photometries)
            mapping_factor = (float(sub_alpha_photometries_max -
                                    sub_alpha_photometries_min) /
                              SAP_optimal_bin_number)
            SAP_hwhm = (alpha_photometry -
                        (i * mapping_factor + sub_alpha_photometries_min))
            break
    alpha_sigma = SAP_hwhm / sqrt(2.0 * log(2.0))
    #compute beta_sigma
    sub_beta_photometries = [log(photometry) for photometry in raw_photometries
                             if 0 < photometry <= beta_photometry]
    min_result, results, concatenated_cost_array, raw_cost_array = \
                 optimal_bin_size_MP(raw_photometries=sub_beta_photometries)
    SBP_optimal_bin_number = min_result[1]
    SBP_hist, SBP_bins = np.histogram(a=sub_beta_photometries,
                                      bins=SBP_optimal_bin_number)
    SBP_hwhm = (beta_photometry - gamma_photometry)
    for (i,), h in sorted(np.ndenumerate(SBP_hist), key=lambda x:x[0][0],
                          reverse=True):
        if h > beta / 2.0:
            continue
        else:
            sub_beta_photometries_min = np.amin(sub_beta_photometries)
            sub_beta_photometries_max = np.amax(sub_beta_photometries)
            mapping_factor = (float(sub_beta_photometries_max -
                                    sub_beta_photometries_min) /
                              SBP_optimal_bin_number)
            SBP_hwhm = ((SBP_hist.shape[0] - i) * mapping_factor +
                        sub_beta_photometries_min)
            break
    beta_sigma = SBP_hwhm / sqrt(2.0 * log(2.0))
    return (optimal_bin_number,
            alpha, alpha_index,
            beta, beta_index,
            gamma, gamma_index,
            alpha_photometry,
            beta_photometry,
            gamma_photometry,
            SAP_optimal_bin_number,
            SAP_hwhm,
            alpha_sigma,
            SBP_optimal_bin_number,
            SBP_hwhm,
            beta_sigma,
            SAP_hist, SAP_bins,
            SBP_hist, SBP_bins)


def fwhm_method_v2(photometries, optimal_bin_number=None):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    raw_photometries = [intensity
                        for channel, cdict in photometries.iteritems()
                        for field, fdict in cdict.iteritems()
                        for (h, w), (category, intensities, row)
                            in fdict.iteritems()
                        for intensity in intensities]
    (optimal_bin_number,
     alpha, alpha_index,
     beta, beta_index,
     gamma, gamma_index,
     alpha_photometry,
     beta_photometry,
     gamma_photometry) = \
                     _get_m0Dm1(raw_photometries=raw_photometries,
                                optimal_bin_number=optimal_bin_number)
    #compute alpha_sigma
    sub_alpha_photometries = [photometry for photometry in raw_photometries
                              if photometry <= alpha_photometry]
    min_result, results, concatenated_cost_array, raw_cost_array = \
                   optimal_bin_size_MP(raw_photometries=sub_alpha_photometries)
    SAP_optimal_bin_number = min_result[1]
    SAP_hist, SAP_bins = np.histogram(a=sub_alpha_photometries,
                                      bins=SAP_optimal_bin_number)
    SAP_hwhm = (gamma_photometry - alpha_photometry) / 2.0
    default_SAP_hwhm = True
    for (i,), h in sorted(np.ndenumerate(SAP_hist), key=lambda x:x[0][0]):
        if h < alpha / 2.0:
            continue
        else:
            sub_alpha_photometries_min = np.amin(sub_alpha_photometries)
            sub_alpha_photometries_max = np.amax(sub_alpha_photometries)
            mapping_factor = (float(sub_alpha_photometries_max -
                                    sub_alpha_photometries_min) /
                              SAP_optimal_bin_number)
            SAP_hwhm = (alpha_photometry -
                        (i * mapping_factor + sub_alpha_photometries_min))
            default_SAP_hwhm = False
            break
    alpha_sigma = SAP_hwhm / sqrt(2.0 * log(2.0))
    #Shift everything by alpha
    adjusted_raw_photometries = [photometry - alpha_photometry
                                 for photometry in raw_photometries]
    adjusted_photometries = {}
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                adjusted_intensities = [intensity - alpha_photometry
                                        for intensity in intensities]
                adjusted_photometries.setdefault(channel, {
                                        }).setdefault(field, {
                                        }).setdefault((h, w),
                        (category, adjusted_intensities, row))
    beta_photometry -= alpha_photometry
    gamma_photometry -= alpha_photometry
    alpha_photometry = 0
    #compute real beta (beta from  _get_m0Dm1 is the mode of the lognormal,
    #which is ***NOT*** its mean.
    #lognormal's mode is e**(mu - sigma**2)
    super_gamma_log_photometries = [log(photometry)
                                    for photometry in adjusted_raw_photometries
                                    if photometry > gamma_photometry]
    min_result, results, concatenated_cost_array, raw_cost_array = \
            optimal_bin_size_MP(raw_photometries=super_gamma_log_photometries,
                                num_processes=None,
                                min_n_bins=10, max_n_bins=10000)
    SGP_optimal_bin_number = min_result[1]
    SGP_hist, SGP_bins = np.histogram(a=super_gamma_log_photometries,
                                      bins=SGP_optimal_bin_number)
    SGP_max, SGP_argmax = np.amax(SGP_hist), np.argmax(SGP_hist)
    if SGP_argmax < len(SGP_hist) - 1:
        SGP_max_logP = np.mean([SGP_bins[SGP_argmax],
                                SGP_bins[SGP_argmax + 1]])
    else:
        SGP_max_logP = SGP_bins[SGP_argmax]
    beta_photometry = math.e**SGP_max_logP
    #get beta_sigma
    SGP_hwhm = abs(SGP_max_logP - log(gamma_photometry)) / 2.0
    default_SGP_hwhm = True
    for (i,), h in sorted(np.ndenumerate(SGP_hist[:SGP_argmax]),
                          key=lambda x:x[0][0],
                          reverse=True):
        if h > SGP_max / 2.0:
            continue
        else:
            SGP_hwhm = SGP_max_logP - np.mean([SGP_bins[i], SGP_bins[i + 1]])
            default_SGP_hwhm = False
            break
    beta_sigma = SGP_hwhm / sqrt(2.0 * log(2.0))
    return (alpha_photometry, alpha_sigma, beta_photometry, beta_sigma,
            adjusted_raw_photometries, adjusted_photometries,
            SAP_hist, SAP_bins, SGP_hist, SGP_bins,
            optimal_bin_number, alpha, alpha_index, beta, beta_index,
            gamma, gamma_index, gamma_photometry,
            default_SAP_hwhm, default_SGP_hwhm)
    


def _intensities_to_signal_lognormal_v3(intensities,
                                        alpha, beta, gamma,
                                        alpha_sigma, beta_sigma,
                                        max_possible=5, allow_multidrop=False,
                                        allow_upsteps=False):
    #Baseline shift everything based on alpha
    beta -= alpha
    gamma -= alpha
    intensities = [intensity - alpha for intensity in intensities]
    #zero_fluor = mu_zero + 2.0 * sigma_zero #This is now gamma
    #one_fluor = mu_one - mu_zero #This is now beta
    two_fluor = 2.0 * beta
    log_one_fluor, log_two_fluor = log(beta), log(two_fluor)
    half_log_fluor = np.mean((log_one_fluor, log_two_fluor)) - log_one_fluor
    #log_fluor_boundaries[1] = upper_log_one
    #log_fluor_boundaries[2] = upper_log_two
    #log_fluor_boundaries[3] = upper_log_three
    #...etc
    #log_fluor_boundaries = [log_one_fluor + (2 * i + 1) * half_log_fluor
    #                        for i in range(max_possible)]
    log_fluor_boundaries = [np.mean([log(beta + i * beta),
                                     log(beta + (i + 1) * beta)])
                            for i in range(max_possible + 1)]
    log_fluor_means = [log(beta + i * beta)
                       for i in range(max_possible + 2)]
    log_max_intensity = log(max(max(intensities), 1))
    lmii = max_possible
    for i, lfb in enumerate(log_fluor_boundaries):
        if log_max_intensity > lfb:
            continue
        else:
            lmii = i + 2
            break
    best_seq = None
    best_score = -1
    log_intensities = [log(intensity) if intensity > 0 else -10000
                       for intensity in intensities]
    best_intensity_scores = None
    if allow_upsteps:
        iterator = product(reversed(range(lmii + 1)),
                           repeat=len(intensities))
    else:
        iterator = combinations_with_replacement(reversed(range(lmii + 1)),
                                                 len(intensities))
    zero_cutoff = (alpha + gamma) / 3.0
    #one_cutoff = (gamma + beta) / 2.0
    for seq in iterator:
        if not allow_multidrop:
            seq_diff = [seq[i] - s for i, s in enumerate(seq[1:])]
            if max(seq_diff) > 1:
                continue
        #if any([(intensity <= zero_cutoff and seq[i] != 0) or
        #        (intensity >= one_cutoff and seq[i] == 0)
        #        for i, intensity in enumerate(intensities)]):
        #    continue
        if any([(intensity <= zero_cutoff and seq[i] != 0)
                for i, intensity in enumerate(intensities)]):
            continue
        #intensity_distances = [abs(log_intensities[i] -
        #                           log_fluor_means[seq[i] - 1]) / beta_sigma
        #                       if seq[i] > 0
        #                       else abs(intensity) / alpha_sigma
        #                       for i, intensity in enumerate(intensities)]
        #if max(intensity_distances) > 4:
        #    continue
        intensity_scores = [norm.pdf(log_intensity,
                                     loc=log_fluor_means[seq[i] - 1],
                                     scale=beta_sigma)
                            if seq[i] > 0
                            else norm.pdf(intensities[i],
                                          loc=0.0, #alpha already adjusted for
                                          scale=alpha_sigma)
                            for i, log_intensity in enumerate(log_intensities)]
        total_score = reduce(mul, intensity_scores, 1.0)
        if total_score > best_score:
            best_seq = seq
            best_score = total_score
            best_intensity_scores = intensity_scores
    if best_seq is not None and best_score > math.e**-13:
        starting_intensity = best_seq[0]
        signal_TFn = [best_seq[f] - fc for f, fc in enumerate(best_seq[1:])]
        signal = []
        for i, tf in enumerate(signal_TFn):
            if tf > 0:
                signal += [('A', i + 1)] * tf
            elif tf < 0:
                signal = None
                break
        if signal is not None:
            signal = tuple(signal)
            if len(signal) == 0:
                signal = [('A', 0)]
            signal = tuple(signal)
            if best_seq[-1] == 0:
                is_zero = True
            else:
                is_zero = False
        else:
            is_zero = None
    else:
        signal = None
        is_zero = None
        starting_intensity = None
    return (signal, is_zero, best_seq, lmii, best_score,
            best_intensity_scores, starting_intensity)


def _photometries_lognormal_fit_MP_v3(photometries, alpha, beta, gamma,
                                      alpha_sigma, beta_sigma,
                                      max_possible=5,
                                      num_processes=None, allow_upsteps=False,
                                      allow_multidrop=False):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                process = pool.apply_async(
                                           _intensities_to_signal_lognormal_v3,
                                               (intensities,
                                                alpha,
                                                beta,
                                                gamma,
                                                alpha_sigma,
                                                beta_sigma,
                                                max_possible,
                                                allow_multidrop,
                                                allow_upsteps))
                processes.append((process, channel, field, h, w, row, category,
                                  intensities))
    pool.close()
    pool.join()
    signals = {}
    none_count = 0
    total_count = 0
    all_fit_info = []
    for i, (process, channel, field, h, w, row, category,
            intensities) in enumerate(processes):
        total_count += 1
        (signal, is_zero, best_seq, lmii, best_score,
         best_intensity_scores, starting_intensity) = process.get()
        all_fit_info.append((channel, field, h, w, row, category, intensities,
                             signal, is_zero, best_seq, lmii, best_score,
                             best_intensity_scores, starting_intensity))
        if signal is None:
            none_count += 1
        else:
            signals.setdefault((signal, is_zero, starting_intensity), 0)
            signals[(signal, is_zero, starting_intensity)] += 1
    return signals, total_count, none_count, all_fit_info


def _intensities_to_signal_lognormal_v4(intensities, #here, already adjusted
                                        alpha, beta, gamma,
                                        alpha_sigma, beta_sigma,
                                        max_possible=5, allow_multidrop=False,
                                        allow_upsteps=False,
                                        lognormal_probability_integral=1.0):
    two_fluor = 2.0 * beta
    log_one_fluor, log_two_fluor = log(beta), log(two_fluor)
    half_log_fluor = np.mean((log_one_fluor, log_two_fluor)) - log_one_fluor
    #log_fluor_boundaries[1] = upper_log_one
    #log_fluor_boundaries[2] = upper_log_two
    #log_fluor_boundaries[3] = upper_log_three
    #...etc
    #log_fluor_boundaries = [log_one_fluor + (2 * i + 1) * half_log_fluor
    #                        for i in range(max_possible)]
    log_fluor_boundaries = [np.mean([log(beta + i * beta),
                                     log(beta + (i + 1) * beta)])
                            for i in range(max_possible + 1)]
    log_fluor_means = [log(beta + i * beta)
                       for i in range(max_possible + 2)]
    log_max_intensity = log(max(max(intensities), 1))
    lmii = max_possible
    for i, lfb in enumerate(log_fluor_boundaries):
        if log_max_intensity > lfb:
            continue
        else:
            lmii = i + 2
            break
    best_seq = None
    best_score = -1
    #log_intensities = [log(intensity) if intensity > 0 else -10000
    #                   for intensity in intensities]
    best_intensity_scores = None
    if allow_upsteps:
        iterator = product(reversed(range(lmii + 1)),
                           repeat=len(intensities))
    else:
        iterator = combinations_with_replacement(reversed(range(lmii + 1)),
                                                 len(intensities))
    zero_cutoff = (alpha + gamma) / 3.0
    #one_cutoff = (gamma + beta) / 2.0
    score_normalization = [norm.pdf(intensity, loc=0.0, scale=alpha_sigma) +
                           sum([lognorm.pdf(intensity, beta_sigma,
                                            loc=0, scale=beta * f)
                                for f in range(1, max_possible + 1)])
                           for i, intensity in enumerate(intensities)]
    maximum_possible_scores_cache = {} #key=seq_value
    intensity_scores_cache = {} #key=(intensity_index, seq_value)
    for seq in iterator:
        if not allow_multidrop:
            seq_diff = [seq[i] - s for i, s in enumerate(seq[1:])]
            if max(seq_diff) > 1:
                continue
        #if any([(intensity <= zero_cutoff and seq[i] != 0) or
        #        (intensity >= one_cutoff and seq[i] == 0)
        #        for i, intensity in enumerate(intensities)]):
        #    continue
        if any([(intensity <= zero_cutoff and seq[i] != 0)
                for i, intensity in enumerate(intensities)]):
            continue
        #intensity_distances = [abs(log_intensities[i] -
        #                           log_fluor_means[seq[i] - 1]) / beta_sigma
        #                       if seq[i] > 0
        #                       else abs(intensity) / alpha_sigma
        #                       for i, intensity in enumerate(intensities)]
        #if max(intensity_distances) > 4:
        #    continue
        #intensity_scores = [#(norm.pdf(log_intensities[i],
        #                    #          loc=log_fluor_means[seq[i] - 1],
        #                    #          scale=beta_sigma) /
        #                    # lognormal_probability_integral)
        #                    lognorm.pdf(intensities[i], beta_sigma, loc=0,
        #                                scale=beta * seq_value)
        #                    if seq_value > 0
        #                    else norm.pdf(intensities[i],
        #                                  loc=0.0, #alpha already adjusted for
        #                                  scale=alpha_sigma)
        #                    for i, seq_value in enumerate(seq)]
        intensity_scores = []
        for i, seq_value in enumerate(seq):
            if (i, seq_value) not in intensity_scores_cache:
                if seq_value == 0:
                    score = norm.pdf(intensities[i], loc=0.0,
                                     scale=alpha_sigma)
                else:
                    score = lognorm.pdf(intensities[i], beta_sigma, loc=0,
                                        scale=beta * seq_value)
                intensity_scores_cache.setdefault((i, seq_value), score)
            intensity_scores.append(intensity_scores_cache[(i, seq_value)])
        intensity_scores = [float(score) / score_normalization[s]
                            for s, score in enumerate(intensity_scores)]
        #maximum_possible_scores = [lognorm.pdf(float(beta) * seq_value /
        #                                       math.e**(beta_sigma**2),
        #                                       beta_sigma,
        #                                       loc=0, scale=beta * seq_value)
        #                           if seq_value > 0
        #                           else norm.pdf(0, loc=0.0,
        #                                         scale=alpha_sigma)
        #                           for seq_value in seq]
        maximum_possible_scores = []
        for seq_value in seq:
            if seq_value not in maximum_possible_scores_cache:
                if seq_value  == 0:
                    score = norm.pdf(0, loc=0.0, scale=alpha_sigma)
                else:
                    score = lognorm.pdf(float(beta) * seq_value /
                                        math.e**(beta_sigma**2),
                                        beta_sigma,
                                        loc=0, scale=beta * seq_value)
                normalization = (norm.pdf(float(beta) * seq_value /
                                          math.e**(beta_sigma**2), loc=0.0,
                                          scale=alpha_sigma) +
                                 sum([lognorm.pdf(float(beta) * seq_value /
                                                  math.e**(beta_sigma**2),
                                                  beta_sigma,
                                                  loc=0, scale=beta * f)
                                      for f in range(1, max_possible + 1)]))
                score /= float(normalization)
                maximum_possible_scores_cache.setdefault(seq_value, score)
            maximum_possible_scores.append(
                                      maximum_possible_scores_cache[seq_value])
        total_score = reduce(mul, intensity_scores, 1.0)
        maximum_possible_score = reduce(mul, maximum_possible_scores, 1.0)
        total_score /= float(maximum_possible_score)
        if total_score > best_score:
            best_seq = seq
            best_score = total_score
            best_intensity_scores = intensity_scores
    if best_seq is not None:
        maximum_possible_score = None
        mpi_score = None
        starting_intensity = best_seq[0]
        signal_TFn = [best_seq[f] - fc for f, fc in enumerate(best_seq[1:])]
        signal = []
        for i, tf in enumerate(signal_TFn):
            if tf > 0:
                signal += [('A', i + 1)] * tf
            elif tf < 0:
                signal = None
                break
        if signal is not None:
            signal = tuple(signal)
            if len(signal) == 0:
                signal = [('A', 0)]
            signal = tuple(signal)
            if best_seq[-1] == 0:
                is_zero = True
            else:
                is_zero = False
        else:
            is_zero = None
    else:
        signal = None
        is_zero = None
        starting_intensity = None
        maximum_possible_score = None
        mpi_score = None
    return (signal, is_zero, best_seq, lmii, best_score,
            best_intensity_scores, starting_intensity, maximum_possible_score,
            mpi_score)


def _photometries_lognormal_fit_MP_v4(photometries, alpha, beta, gamma,
                                      alpha_sigma, beta_sigma,
                                      max_possible=5,
                                      num_processes=None, allow_upsteps=False,
                                      allow_multidrop=False):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    #Need to calculate normalization integral for the lognormal probabilities
    #lognormal_probability_integral = 0
    #subinterval_area = 10**10
    #value = 1
    #while value < beta or subinterval_area > 0.001:
    #    #Note that area has implicit base 1 since delta_value = 1
    #    subinterval_area = norm.pdf(log(value), loc=log(beta),
    #                                scale=beta_sigma)
    #    lognormal_probability_integral += subinterval_area
    #    value += 1
    #lognormal_probability_integral = float(lognormal_probability_integral)
    lognormal_probability_integral = None
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                process = pool.apply_async(
                                          _intensities_to_signal_lognormal_v4,
                                              (intensities,
                                               alpha,
                                               beta,
                                               gamma,
                                               alpha_sigma,
                                               beta_sigma,
                                               max_possible,
                                               allow_multidrop,
                                               allow_upsteps,
                                               lognormal_probability_integral))
                processes.append((process, channel, field, h, w, row, category,
                                  intensities))
    pool.close()
    pool.join()
    signals = {}
    none_count = 0
    total_count = 0
    all_fit_info = []
    for i, (process, channel, field, h, w, row, category,
            intensities) in enumerate(processes):
        total_count += 1
        (signal, is_zero, best_seq, lmii, best_score,
         best_intensity_scores, starting_intensity,
         max_possible_score, mpi_score) = process.get()
        all_fit_info.append((channel, field, h, w, row, category, intensities,
                             signal, is_zero, best_seq, lmii, best_score,
                             best_intensity_scores, starting_intensity,
                             max_possible_score, mpi_score))
        if signal is None:
            none_count += 1
        else:
            signals.setdefault((signal, is_zero, starting_intensity), 0)
            signals[(signal, is_zero, starting_intensity)] += 1
    return (signals, total_count, none_count, all_fit_info,
            lognormal_probability_integral)


def _intensities_to_signal_lognormal_v5(intensities, #here, already adjusted
                                        alpha, beta, gamma,
                                        alpha_sigma, beta_sigma,
                                        max_possible=5, allow_multidrop=False,
                                        allow_upsteps=False,
                                        max_deviation=3,
                                        quench_factor=0):
    #log_fluor_boundaries[0] = upper_log_one
    #log_fluor_boundaries[1] = upper_log_two
    #log_fluor_boundaries[2] = upper_log_three
    #...etc
    log_fluor_boundaries = [np.mean([log(beta) + log(i + 1.0) -
                                     quench_factor * max(i - 1, 0),
                                     log(beta) + log(i + 2.0) -
                                     quench_factor * i, 0])
                            for i in range(max_possible + 1)]
    log_fluor_means = [log(beta) + log(i + 1.0) - quench_factor * max(i - 1, 0)
                       for i in range(max_possible + 2)]
    log_max_intensity = log(max(max(intensities), 1))
    lmii = max_possible
    for i, lfb in enumerate(log_fluor_boundaries):
        if log_max_intensity > lfb:
            continue
        else:
            lmii = i + 2
            break
    best_seq = None
    best_score = -1
    log_intensities = [log(intensity) if intensity > 0 else -10000
                       for intensity in intensities]
    best_intensity_scores = None
    if allow_upsteps:
        iterator = product(reversed(range(lmii + 1)),
                           repeat=len(intensities))
    else:
        iterator = combinations_with_replacement(reversed(range(lmii + 1)),
                                                 len(intensities))
    zero_cutoff = (alpha + gamma) / 3.0
    intensity_scores_cache = {} #key=(intensity_index, seq_value)
    sigma_ratio = float(alpha_sigma) / beta_sigma
    for seq in iterator:
        if not allow_multidrop:
            seq_diff = [seq[i] - s for i, s in enumerate(seq[1:])]
            if max(seq_diff) > 1:
                continue
        if any([(intensity <= zero_cutoff and seq[i] != 0)
                for i, intensity in enumerate(intensities)]):
            continue
        intensity_deviations = [(abs(log_intensities[i] -
                                     log_fluor_means[seq_value - 1]) /
                                 beta_sigma)
                                if seq_value > 0
                                else abs(intensities[i]) / alpha_sigma
                                for i, seq_value in enumerate(seq)]
        if max(intensity_deviations) > max_deviation:
            continue
        intensity_scores = []
        for i, seq_value in enumerate(seq):
            if (i, seq_value) not in intensity_scores_cache:
                if seq_value == 0:
                    score = norm.pdf(intensities[i] / sigma_ratio,
                                     loc=0.0,
                                     scale=beta_sigma)
                else:
                    score = norm.pdf(log_intensities[i],
                                     loc=log_fluor_means[seq_value - 1],
                                     scale=beta_sigma)
                intensity_scores_cache.setdefault((i, seq_value), score)
            intensity_scores.append(intensity_scores_cache[(i, seq_value)])
        total_score = reduce(mul, intensity_scores, 1.0)
        if total_score > best_score:
            best_seq = seq
            best_score = total_score
            best_intensity_scores = intensity_scores
    if best_seq is not None:
        starting_intensity = best_seq[0]
        signal_TFn = [best_seq[f] - fc for f, fc in enumerate(best_seq[1:])]
        signal = []
        for i, tf in enumerate(signal_TFn):
            if tf > 0:
                signal += [('A', i + 1)] * tf
            elif tf < 0:
                signal = None
                break
        if signal is not None:
            signal = tuple(signal)
            if len(signal) == 0:
                signal = [('A', 0)]
            signal = tuple(signal)
            if best_seq[-1] == 0:
                is_zero = True
            else:
                is_zero = False
        else:
            is_zero = None
    else:
        signal = None
        is_zero = None
        starting_intensity = None
    return (signal, is_zero, best_seq, lmii, best_score,
            best_intensity_scores, starting_intensity)


def _photometries_lognormal_fit_MP_v5(photometries, alpha, beta, gamma,
                                      alpha_sigma, beta_sigma,
                                      max_possible=5,
                                      num_processes=None, allow_upsteps=False,
                                      allow_multidrop=False,
                                      max_deviation=3,
                                      quench_factor=0):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                process = pool.apply_async(
                                          _intensities_to_signal_lognormal_v5,
                                              (intensities,
                                               alpha,
                                               beta,
                                               gamma,
                                               alpha_sigma,
                                               beta_sigma,
                                               max_possible,
                                               allow_multidrop,
                                               allow_upsteps,
                                               max_deviation,
                                               quench_factor))
                processes.append((process, channel, field, h, w, row, category,
                                  intensities))
    pool.close()
    pool.join()
    signals = {}
    none_count = 0
    total_count = 0
    all_fit_info = []
    for i, (process, channel, field, h, w, row, category,
            intensities) in enumerate(processes):
        total_count += 1
        (signal, is_zero, best_seq, lmii, best_score,
         best_intensity_scores, starting_intensity) = process.get()
        all_fit_info.append((channel, field, h, w, row, category, intensities,
                             signal, is_zero, best_seq, lmii, best_score,
                             best_intensity_scores, starting_intensity))
        if signal is None:
            none_count += 1
        else:
            signals.setdefault((signal, is_zero, starting_intensity), 0)
            signals[(signal, is_zero, starting_intensity)] += 1
    return signals, total_count, none_count, all_fit_info


def _intensities_to_signal_lognormal_v6(intensities, #here, already adjusted
                                        alpha, beta, gamma,
                                        alpha_sigma, beta_sigma,
                                        max_possible=5, allow_multidrop=False,
                                        allow_upsteps=False,
                                        max_deviation=3,
                                        quench_factor=0, 
                                        deltas=None, gamma_score=None):
    #log_fluor_boundaries[0] = upper_log_one
    #log_fluor_boundaries[1] = upper_log_two
    #log_fluor_boundaries[2] = upper_log_three
    #...etc
    log_fluor_boundaries = [np.mean([log(beta) + log(i + 1.0) -
                                     quench_factor * max(i - 1, 0),
                                     log(beta) + log(i + 2.0) -
                                     quench_factor * i, 0])
                            for i in range(max_possible + 1)]
    log_fluor_means = [log(beta) + log(i + 1.0) - quench_factor * max(i - 1, 0)
                       for i in range(max_possible + 2)]
    log_max_intensity = log(max(max(intensities), 1))
    lmii = max_possible
    for i, lfb in enumerate(log_fluor_boundaries):
        if log_max_intensity > lfb:
            continue
        else:
            lmii = i + 2
            break
    best_seq = None
    best_score = -1
    log_intensities = [log(intensity) if intensity > 0 else -10000
                       for intensity in intensities]
    best_intensity_scores = None
    if allow_upsteps:
        iterator = product(reversed(range(lmii + 1)),
                           repeat=len(intensities))
    else:
        iterator = combinations_with_replacement(reversed(range(lmii + 1)),
                                                 len(intensities))
    zero_cutoff = (alpha + gamma) / 3.0
    intensity_scores_cache = {} #key=(intensity_index, seq_value)
    sigma_ratio = float(alpha_sigma) / beta_sigma
    if deltas is not None:
        delta_0, delta_1 = deltas
        gamma_score *= norm.pdf(0, loc=0, scale=beta_sigma)
    for seq in iterator:
        if not allow_multidrop:
            seq_diff = [seq[i] - s for i, s in enumerate(seq[1:])]
            if max(seq_diff) > 1:
                continue
        if any([(intensity <= zero_cutoff and seq[i] != 0)
                for i, intensity in enumerate(intensities)]):
            continue
        intensity_deviations = [(abs(log_intensities[i] -
                                     log_fluor_means[seq_value - 1]) /
                                 beta_sigma)
        #                        else abs(intensities[i]) / alpha_sigma
                                for i, seq_value in enumerate(seq)
				if seq_value > 0]
        if (len(intensity_deviations) > 0 and
            max(intensity_deviations) > max_deviation):
            continue
        #special intensity_deviations case for 0 when deltas are given
        over_deviation = True
        for i, seq_value in enumerate(seq):
            if seq_value > 0:
                continue
            if (deltas is None and
                abs(intensities[i]) / alpha_sigma > max_deviation):
                break
            elif (deltas is not None and
                  not delta_0 <= intensities[i] <= delta_1 and
                  abs(intensities[i]) / alpha_sigma > max_deviation):
                break
        else:
            over_deviation = False
        if over_deviation:
            continue
        intensity_scores = []
        for i, seq_value in enumerate(seq):
            if (i, seq_value) not in intensity_scores_cache:
                if seq_value == 0:
                    if (deltas is not None and
                        delta_0 <= intensities[i] <= delta_1):
                        score = gamma_score
                    else:
                        score = norm.pdf(intensities[i] / sigma_ratio,
                                         loc=0.0,
                                         scale=beta_sigma)
                else:
                    score = norm.pdf(log_intensities[i],
                                     loc=log_fluor_means[seq_value - 1],
                                     scale=beta_sigma)
                intensity_scores_cache.setdefault((i, seq_value), score)
            intensity_scores.append(intensity_scores_cache[(i, seq_value)])
        total_score = reduce(mul, intensity_scores, 1.0)
        if total_score > best_score:
            best_seq = seq
            best_score = total_score
            best_intensity_scores = intensity_scores
    if best_seq is not None:
        starting_intensity = best_seq[0]
        signal_TFn = [best_seq[f] - fc for f, fc in enumerate(best_seq[1:])]
        signal = []
        for i, tf in enumerate(signal_TFn):
            if tf > 0:
                signal += [('A', i + 1)] * tf
            elif tf < 0:
                signal = None
                break
        if signal is not None:
            signal = tuple(signal)
            if len(signal) == 0:
                signal = [('A', 0)]
            signal = tuple(signal)
            if best_seq[-1] == 0:
                is_zero = True
            else:
                is_zero = False
        else:
            is_zero = None
    else:
        signal = None
        is_zero = None
        starting_intensity = None
    return (signal, is_zero, best_seq, lmii, best_score,
            best_intensity_scores, starting_intensity)


def _find_deltas(alpha_sigma, beta, beta_sigma, gamma_score):
    sigma_ratio = float(alpha_sigma) / beta_sigma
    f0_distribution = norm(loc=0, scale=beta_sigma) #scale by sigma_ratio
    f1_distribution = norm(loc=log(beta), scale=beta_sigma)
    delta_0, delta_1 = None, None
    for photometry in range(1, int(math.ceil(beta)) + 1):
        f0_value = f0_distribution.pdf(photometry / sigma_ratio)
        f1_value = f1_distribution.pdf(log(photometry))
        if delta_0 is None and f0_value < gamma_score:
            delta_0 = photometry
        if delta_0 is not None and delta_1 is None and f1_value > gamma_score:
            delta_1 = photometry
        if delta_0 is not None and delta_1 is not None:
            break
    return delta_0, delta_1


def _photometries_lognormal_fit_MP_v6(photometries, alpha, beta, gamma,
                                      alpha_sigma, beta_sigma,
                                      max_possible=5,
                                      num_processes=None, allow_upsteps=False,
                                      allow_multidrop=False,
                                      max_deviation=3,
                                      quench_factor=0, gamma_score=None):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    deltas = _find_deltas(alpha_sigma=alpha_sigma, beta=beta,
                          beta_sigma=beta_sigma, gamma_score=gamma_score)
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                process = pool.apply_async(
                                          _intensities_to_signal_lognormal_v6,
                                              (intensities,
                                               alpha,
                                               beta,
                                               gamma,
                                               alpha_sigma,
                                               beta_sigma,
                                               max_possible,
                                               allow_multidrop,
                                               allow_upsteps,
                                               max_deviation,
                                               quench_factor,
                                               deltas,
                                               gamma_score))
                processes.append((process, channel, field, h, w, row, category,
                                  intensities))
    pool.close()
    pool.join()
    signals = {}
    none_count = 0
    total_count = 0
    all_fit_info = []
    for i, (process, channel, field, h, w, row, category,
            intensities) in enumerate(processes):
        total_count += 1
        (signal, is_zero, best_seq, lmii, best_score,
         best_intensity_scores, starting_intensity) = process.get()
        all_fit_info.append((channel, field, h, w, row, category, intensities,
                             signal, is_zero, best_seq, lmii, best_score,
                             best_intensity_scores, starting_intensity))
        if signal is None:
            none_count += 1
        else:
            signals.setdefault((signal, is_zero, starting_intensity), 0)
            signals[(signal, is_zero, starting_intensity)] += 1
    return signals, total_count, none_count, all_fit_info, deltas


def _intensities_to_signal_lognormal_v7(intensities, #here, already adjusted
                                        alpha, beta, gamma,
                                        alpha_sigma, beta_sigma,
                                        max_possible=5, allow_multidrop=False,
                                        allow_upsteps=False,
                                        max_deviation=3,
                                        quench_factor=0, 
                                        deltas=None, gamma_score=None,
                                        categories=None):
    if categories is None:
        raise ValueError("categories required in v7")
    #log_fluor_boundaries[0] = upper_log_one
    #log_fluor_boundaries[1] = upper_log_two
    #log_fluor_boundaries[2] = upper_log_three
    #...etc
    log_fluor_boundaries = [np.mean([log(beta) + log(i + 1.0) -
                                     quench_factor * max(i - 1, 0),
                                     log(beta) + log(i + 2.0) -
                                     quench_factor * i, 0])
                            for i in range(max_possible + 1)]
    log_fluor_means = [log(beta) + log(i + 1.0) - quench_factor * max(i - 1, 0)
                       for i in range(max_possible + 2)]
    log_max_intensity = log(max(max(intensities), 1))
    lmii = max_possible
    for i, lfb in enumerate(log_fluor_boundaries):
        if log_max_intensity > lfb:
            continue
        else:
            lmii = i + 2
            break
    best_seq = None
    best_score = -1
    log_intensities = [log(intensity) if intensity > 0 else -10000
                       for intensity in intensities]
    best_intensity_scores = None
    if allow_upsteps:
        iterator = product(reversed(range(lmii + 1)),
                           repeat=len(intensities))
    else:
        iterator = combinations_with_replacement(reversed(range(lmii + 1)),
                                                 len(intensities))
    #zero_cutoff = (alpha + gamma) / 3.0
    intensity_scores_cache = {} #key=(intensity_index, seq_value)
    sigma_ratio = float(alpha_sigma) / beta_sigma
    if deltas is not None:
        raise DeprecationWarning("v7 doesn't use deltas")
        delta_0, delta_1 = deltas
        gamma_score *= norm.pdf(0, loc=0, scale=beta_sigma)
    for seq in iterator:
        if any([((categories[i] and seq_value == 0) or
                 (not categories[i] and seq_value > 0))
                for i, seq_value in enumerate(seq)]):
            continue
        if not allow_multidrop:
            seq_diff = [seq[i] - s for i, s in enumerate(seq[1:])]
            if max(seq_diff) > 1:
                continue
        #if any([(intensity <= zero_cutoff and seq[i] != 0)
        #        for i, intensity in enumerate(intensities)]):
        #    continue
        intensity_deviations = [(abs(log_intensities[i] -
                                     log_fluor_means[seq_value - 1]) /
                                 beta_sigma)
        #                        else abs(intensities[i]) / alpha_sigma
                                for i, seq_value in enumerate(seq)
				if seq_value > 0]
        if (len(intensity_deviations) > 0 and
            max(intensity_deviations) > max_deviation):
            continue
        #special intensity_deviations case for 0 when deltas are given
        #over_deviation = True
        #for i, seq_value in enumerate(seq):
        #    if seq_value > 0:
        #        continue
        #    if (deltas is None and
        #        abs(intensities[i]) / alpha_sigma > max_deviation):
        #        break
        #    elif (deltas is not None and
        #          not delta_0 <= intensities[i] <= delta_1 and
        #          abs(intensities[i]) / alpha_sigma > max_deviation):
        #        break
        #else:
        #    over_deviation = False
        #if over_deviation:
        #    continue
        intensity_scores = []
        for i, seq_value in enumerate(seq):
            if (i, seq_value) not in intensity_scores_cache:
                if seq_value == 0:
                    #if (deltas is not None and
                    #    delta_0 <= intensities[i] <= delta_1):
                    #    score = gamma_score
                    #else:
                    #    score = norm.pdf(intensities[i] / sigma_ratio,
                    #                     loc=0.0,
                    #                     scale=beta_sigma)
                    score = 1.0
                else:
                    score = norm.pdf(log_intensities[i],
                                     loc=log_fluor_means[seq_value - 1],
                                     scale=beta_sigma)
                intensity_scores_cache.setdefault((i, seq_value), score)
            intensity_scores.append(intensity_scores_cache[(i, seq_value)])
        total_score = reduce(mul, intensity_scores, 1.0)
        if total_score > best_score:
            best_seq = seq
            best_score = total_score
            best_intensity_scores = intensity_scores
    if best_seq is not None:
        starting_intensity = best_seq[0]
        signal_TFn = [best_seq[f] - fc for f, fc in enumerate(best_seq[1:])]
        signal = []
        for i, tf in enumerate(signal_TFn):
            if tf > 0:
                signal += [('A', i + 1)] * tf
            elif tf < 0:
                signal = None
                break
        if signal is not None:
            signal = tuple(signal)
            if len(signal) == 0:
                signal = [('A', 0)]
            signal = tuple(signal)
            if best_seq[-1] == 0:
                is_zero = True
            else:
                is_zero = False
        else:
            is_zero = None
    else:
        signal = None
        is_zero = None
        starting_intensity = None
    return (signal, is_zero, best_seq, lmii, best_score,
            best_intensity_scores, starting_intensity)


def _photometries_lognormal_fit_MP_v7(photometries, alpha, beta, gamma,
                                      alpha_sigma, beta_sigma,
                                      max_possible=5,
                                      num_processes=None, allow_upsteps=False,
                                      allow_multidrop=False,
                                      max_deviation=3,
                                      quench_factor=0, gamma_score=None):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    deltas = _find_deltas(alpha_sigma=alpha_sigma, beta=beta,
                          beta_sigma=beta_sigma, gamma_score=gamma_score)
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                process = pool.apply_async(
                                          _intensities_to_signal_lognormal_v7,
                                              (intensities,
                                               alpha,
                                               beta,
                                               gamma,
                                               alpha_sigma,
                                               beta_sigma,
                                               max_possible,
                                               allow_multidrop,
                                               allow_upsteps,
                                               max_deviation,
                                               quench_factor,
                                               None, #deltas,
                                               gamma_score,
                                               category))
                processes.append((process, channel, field, h, w, row, category,
                                  intensities))
    pool.close()
    pool.join()
    signals = {}
    none_count = 0
    total_count = 0
    all_fit_info = []
    for i, (process, channel, field, h, w, row, category,
            intensities) in enumerate(processes):
        total_count += 1
        (signal, is_zero, best_seq, lmii, best_score,
         best_intensity_scores, starting_intensity) = process.get()
        all_fit_info.append((channel, field, h, w, row, category, intensities,
                             signal, is_zero, best_seq, lmii, best_score,
                             best_intensity_scores, starting_intensity))
        if signal is None:
            none_count += 1
        else:
            signals.setdefault((signal, is_zero, starting_intensity), 0)
            signals[(signal, is_zero, starting_intensity)] += 1
    return signals, total_count, none_count, all_fit_info, deltas


def last_drop_method(photometries):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    last_drop_list = [log(iON - iOFF)
                      for channel, cdict in photometries.iteritems()
                      for field, fdict in cdict.iteritems()
                      for (h, w), (category, intensities, row) in fdict.iteritems()
                      for i, (iON, iOFF) in enumerate(_pairwise(intensities))
                      if category[i] and not category[i + 1] and iON > iOFF]
    obn = optimal_bin_size_MP(last_drop_list)[0][1]
    hist, bins = np.histogram(a=last_drop_list, bins=obn)
    hist_max, hist_argmax = np.amax(hist), np.argmax(hist)
    if hist_argmax < len(bins) - 1:
        hist_max_logP = np.mean([bins[hist_argmax], bins[hist_argmax + 1]])
    else:
        hist_max_logP = bins[hist_argmax]
    hwhm = hist_max_logP / 2.0
    for (i,), h in sorted(np.ndenumerate(hist[:hist_argmax]),
                          key=lambda x:x[0][0], reverse=True):
        if h > hist_max / 2.0:
            continue
        else:
            hwhm = hist_max_logP - np.mean([bins[i], bins[i + 1]])
            break
    beta = math.e**hist_max_logP
    beta_sigma = hwhm / sqrt(2.0 * log(2.0))
    return beta, beta_sigma


def last_drop_method_v2(photometries):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    last_drop_list = [log(iON)
                      for channel, cdict in photometries.iteritems()
                      for field, fdict in cdict.iteritems()
                      for (h, w), (category, intensities, row) in fdict.iteritems()
                      for i, (iON, iOFF) in enumerate(_pairwise(intensities))
                      if category[i] and not category[i + 1] and iON > 0]
    obn = optimal_bin_size_MP(last_drop_list)[0][1]
    hist, bins = np.histogram(a=last_drop_list, bins=obn)
    hist_max, hist_argmax = np.amax(hist), np.argmax(hist)
    if hist_argmax < len(bins) - 1:
        hist_max_logP = np.mean([bins[hist_argmax], bins[hist_argmax + 1]])
    else:
        hist_max_logP = bins[hist_argmax]
    hwhm = hist_max_logP / 2.0
    for (i,), h in sorted(np.ndenumerate(hist[:hist_argmax]),
                          key=lambda x:x[0][0], reverse=True):
        if h > hist_max / 2.0:
            continue
        else:
            hwhm = hist_max_logP - np.mean([bins[i], bins[i + 1]])
            break
    beta = math.e**hist_max_logP
    beta_sigma = hwhm / sqrt(2.0 * log(2.0))
    return beta, beta_sigma


def _intensities_to_signal_lognormal_v8(intensities, #here, already adjusted
                                        beta, beta_sigma,
                                        max_possible=5, allow_multidrop=True,
                                        allow_upsteps=False,
                                        max_deviation=3,
                                        quench_factor=0, 
                                        categories=None,
                                        log_fluor_boundaries=None,
                                        log_fluor_means=None):
    if categories is None:
        raise ValueError("categories required in v7+")
    #log_fluor_boundaries[0] = upper_log_one
    #log_fluor_boundaries[1] = upper_log_two
    #log_fluor_boundaries[2] = upper_log_three
    #...etc
    if log_fluor_boundaries is None:
        log_fluor_boundaries = [np.mean([log(beta) + log(i + 1.0) -
                                         quench_factor * max(i - 1, 0),
                                         log(beta) + log(i + 2.0) -
                                         quench_factor * i, 0])
                                for i in range(max_possible + 1)]
    if log_fluor_means is None:
        raise ValueError("v8+ requires log_fluor_means to be passed manually")
        log_fluor_means = [log(beta) + log(i + 1.0) -
                           quench_factor * max(i - 1, 0)
                           for i in range(max_possible + 2)]
    log_max_intensity = log(max(max(intensities), 1))
    lmii = max_possible
    #for i, lfb in enumerate(log_fluor_boundaries):
    #    if log_max_intensity > lfb:
    #        continue
    #    else:
    #        lmii = i + 2
    #        break
    best_seq = None
    best_score = -1
    log_intensities = [log(intensity) if intensity > 0 else -10000
                       for intensity in intensities]
    best_intensity_scores = None
    if allow_upsteps:
        iterator = product(reversed(range(lmii + 1)),
                           repeat=len(intensities))
    else:
        iterator = combinations_with_replacement(reversed(range(lmii + 1)),
                                                 len(intensities))
    intensity_scores_cache = {} #key=(intensity_index, seq_value)
    norm_function_cache = {i: norm(loc=log_fluor_means[i], scale=beta_sigma)
                           for i in range(lmii + 1)}
    for seq in iterator:
        if any([((categories[i] and seq_value == 0) or
                 (not categories[i] and seq_value > 0))
                for i, seq_value in enumerate(seq)]):
            continue
        if not allow_multidrop:
            seq_diff = [seq[i] - s for i, s in enumerate(seq[1:])]
            if max(seq_diff) > 1:
                continue
        intensity_deviations = [(abs(log_intensities[i] -
                                     log_fluor_means[seq_value - 1]) /
                                 beta_sigma)
                                for i, seq_value in enumerate(seq)
				if seq_value > 0]
        if (len(intensity_deviations) > 0 and
            max(intensity_deviations) > max_deviation):
            continue
        intensity_scores = []
        for i, seq_value in enumerate(seq):
            if (i, seq_value) not in intensity_scores_cache:
                if seq_value == 0:
                    score = 1.0
                else:
                    score = norm_function_cache[seq_value - 1].pdf(
                                                          x=log_intensities[i])
                intensity_scores_cache.setdefault((i, seq_value), score)
            intensity_scores.append(intensity_scores_cache[(i, seq_value)])
        total_score = reduce(mul, intensity_scores, 1.0)
        if total_score > best_score:
            best_seq = seq
            best_score = total_score
            best_intensity_scores = intensity_scores
    if best_seq is not None:
        starting_intensity = best_seq[0]
        signal_TFn = [best_seq[f] - fc for f, fc in enumerate(best_seq[1:])]
        signal = []
        for i, tf in enumerate(signal_TFn):
            if tf > 0:
                signal += [('A', i + 1)] * tf
            elif tf < 0:
                signal = None
                break
        if signal is not None:
            signal = tuple(signal)
            if len(signal) == 0:
                signal = [('A', 0)]
            signal = tuple(signal)
            if best_seq[-1] == 0:
                is_zero = True
            else:
                is_zero = False
        else:
            is_zero = None
    else:
        signal = None
        is_zero = None
        starting_intensity = None
    return (signal, is_zero, best_seq, lmii, best_score,
            best_intensity_scores, starting_intensity)


def _photometries_lognormal_fit_MP_v8(photometries, beta, beta_sigma,
                                      max_possible=5,
                                      num_processes=None, allow_upsteps=False,
                                      allow_multidrop=True,
                                      max_deviation=3,
                                      quench_factor=0,
                                      quench_factors=None):
    if len(photometries) > 1:
        raise NotImplementedError("Currently puts all photometries together, "
                                  "can't handle multiple channels at once.")
    if quench_factors is None or len(quench_factors) != max_possible + 2:
        raise ValueError("quench_factors required for v8+")
    log_fluor_boundaries = [np.mean([log(beta) + log(i + 1.0) -
                                     quench_factor * max(i - 1, 0),
                                     log(beta) + log(i + 2.0) -
                                     quench_factor * i, 0])
                            for i in range(max_possible + 1)]
    #quench_factors[0] must be 0 in sane situation
    #quench_factors[1] is for second fluor, etc
    log_fluor_means = [log(beta) + log(i + 1.0) - quench_factors[i]
                       for i in range(max_possible + 2)]
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = []
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                process = pool.apply_async(
                                          _intensities_to_signal_lognormal_v8,
                                              (intensities,
                                               beta,
                                               beta_sigma,
                                               max_possible,
                                               allow_multidrop,
                                               allow_upsteps,
                                               max_deviation,
                                               quench_factor,
                                               category,
                                               log_fluor_boundaries,
                                               log_fluor_means))
                processes.append((process, channel, field, h, w, row, category,
                                  intensities))
    pool.close()
    pool.join()
    signals = {}
    none_count = 0
    total_count = 0
    all_fit_info = []
    for i, (process, channel, field, h, w, row, category,
            intensities) in enumerate(processes):
        total_count += 1
        (signal, is_zero, best_seq, lmii, best_score,
         best_intensity_scores, starting_intensity) = process.get()
        all_fit_info.append((channel, field, h, w, row, category, intensities,
                             signal, is_zero, best_seq, lmii, best_score,
                             best_intensity_scores, starting_intensity))
        if signal is None:
            none_count += 1
        else:
            signals.setdefault((signal, is_zero, starting_intensity), 0)
            signals[(signal, is_zero, starting_intensity)] += 1
    return signals, total_count, none_count, all_fit_info

def unwind_photometries(photometries):
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                yield (channel, field, h, w, category, intensities, row)

def write_photometries_dict_to_csv(photometries, filepath, dialect='excel'):
    output_writer = csv.writer(open(filepath, 'w'), dialect=dialect)
    cdict = next(photometries.itervalues())
    fdict = next(cdict.itervalues())
    category, intensities, row = next(fdict.itervalues())
    num_cycles = len(category)
    header = (
              ['CHANNEL', 'FIELD', 'H', 'W', 'CATEGORY']
              + ['FRAME ' + str(i) for i in range(num_cycles)]
             )
    output_writer.writerow(header)
    row_counter = 0
    for (channel, field, h, w,
         category, intensities, row) in unwind_photometries(photometries):
        row = (
               [str(channel), str(field), str(h), str(w), str(category)]
               + [str(intensity) for intensity in intensities]
              )
        output_writer.writerow(row)
        row_counter += 1
    return row_counter


def is_multidrop(signal):
    positions = [pos for aa, pos in signal]
    if len(positions) == len(set(positions)):
        is_multidrop = False
    elif len(positions) > len(set(positions)):
        is_multidrop = True
    else:
        raise Exception()
    return is_multidrop


def discard_late_signals(signals,
                         max_cycle=None,
                        ):
    if max_cycle is None:
        filtered_signals = dict(signals)
    else:
        filtered_signals = {}
        for (s, z, si), count in signals.iteritems():
            drop_positions = [pos for aa, pos in s]
            latest_drop_position = max(drop_positions)
            if latest_drop_position > max_cycle:
                continue
            else:
                filtered_signals.setdefault((s, z, si), count)
    return filtered_signals


def head_truncate(signals,
                  num_cycles=None,
                 ):
    if num_cycles is None or num_cycles == 0:
        truncated_signals = dict(signals)
    elif num_cycles > 0:
        truncated_signals = {}
        for (s, z, si), f in signals.iteritems():
            earliest_position = min([pos for aa, pos in s])
            if earliest_position <= num_cycles:
                continue
            shifted_s = tuple([(aa, pos - num_cycles) for aa, pos in s])
            truncated_signals.setdefault((shifted_s, z, si), f)
    else:
        raise ValueError("num_cycles must be None or a non-negative integer.")
    return truncated_signals


def counts_to_percent(signals,
                      include_remainders=False,
                      include_multidrop=True,
                      max_cycle=None,
                     ):
    filtered_signals = {(s, z, si): count
                        for (s, z, si), count in signals.iteritems()
                        if (include_remainders
                            or (not include_remainders and z)
                           )
                       }
    filtered_signals = {(s, z, si): count
                        for (s, z, si), count in filtered_signals.iteritems()
                        if (include_multidrop
                            or (not include_multidrop and not is_multidrop(s))
                           )
                       }
    filtered_signals = discard_late_signals(signals=filtered_signals,
                                            max_cycle=max_cycle,
                                           )
    total_count = sum([count
                       for (s, z, si), count in filtered_signals.iteritems()])
    percent_signals = {(s, z, si): float(count) / total_count
                       for (s, z, si), count in filtered_signals.iteritems()}
    return percent_signals


def sum_signals(experiments):
    #not using defaultdict because want to keep entries int or float contingent
    #on input
    #summed_signals = defaultdict(float)
    summed_signals = {}
    for signals in experiments:
        for (s, z, si), num in signals.iteritems():
            summed_signals.setdefault((s, z, si), 0)
            summed_signals[(s, z, si)] += num
    return summed_signals

def average_signals(experiments,
                    include_remainders=False,
                    include_multidrop=True,
                    max_cycle=None,
                   ):
    """experiments is list of signals dictionaries to be averaged"""
    experiments_percent = [
                       counts_to_percent(signals=signals,
                                         include_remainders=include_remainders,
                                         include_multidrop=include_multidrop,
                                         max_cycle=max_cycle,
                                        )
                           for signals in experiments]
    combined_keys = tuple(set([(s, z, si)
                               for signals in experiments_percent
                               for (s, z, si) in signals]))
    summed_signals = sum_signals(experiments_percent)
    averaged_signals = {(s, z, si): float(summed_signals[(s, z, si)])
                                    / len(experiments)
                        for (s, z, si) in combined_keys}
    return averaged_signals


def signals_std(experiments,
                include_remainders=False,
                include_multidrop=True,
                max_cycle=None,
               ):
    """experiments is list of signals dictionaries to be std'd"""
    experiments_percent = [
                       counts_to_percent(signals=signals,
                                         include_remainders=include_remainders,
                                         include_multidrop=include_multidrop,
                                         max_cycle=max_cycle,
                                        )
                           for signals in experiments]
    combined_ledger = defaultdict(list)
    combined_keys = tuple(set([(s, z, si)
                               for signals in experiments_percent
                               for (s, z, si) in signals]))
    for percents in experiments_percent:
        for (s, z, si) in combined_keys:
            num = percents.get((s, z, si), 0)
            combined_ledger[(s, z, si)].append(num)
    stds = {(s, z, si): np.std(ledger)
            for (s, z, si), ledger in combined_ledger.iteritems()}
    return stds


def generate_adjacent_positions(signal,
                                include_multidrop=False,
                               ):
    if len(signal) == 0:
        raise ValueError("Not defined for empty signal.")
    if not signal[1]:
        raise ValueError("Not defined for remainders.")
    amino_acid_set = set([aa for aa, pos in signal[0]])
    if len(amino_acid_set) != 1:
        raise ValueError("Currently only implemented for one label.")
    positions = tuple([pos for aa, pos in signal[0]])
    position_perturbations = product((-1, 0, 1), repeat=len(positions))
    adjacent_positions = []
    for perturbation in position_perturbations:
        if all([p == 0 for p in perturbation]):
            continue
        perturbed_positions = [pos + perturbation[p]
                               for p, pos in enumerate(positions)]
        if (not include_multidrop
            and len(set(perturbed_positions)) < len(perturbed_positions)):
            continue
        adjacent_positions.append(tuple(perturbed_positions))
    return adjacent_positions


def interpolate_signal(signals,
                       interpolation_target,
                       num_cycles,
                       include_multidrop=False,
                      ):
    amino_acid_set = set([aa for signal in signals for aa, pos in signal[0]])
    if len(amino_acid_set) != 1:
        raise ValueError("Currently only implemented for one label.")
    else:
        used_amino_acid = amino_acid_set.pop()
    adjacent_positions = generate_adjacent_positions(
                                           signal=interpolation_target,
                                           include_multidrop=include_multidrop,
                                                    )
    adjacent_signals = [(tuple([(used_amino_acid, pos) for pos in adj]),
                         interpolation_target[1],
                         interpolation_target[2])
                        for adj in adjacent_positions
                        if all([0 < pos <= num_cycles for pos in adj])]
    adjacent_values = {s: signals.get(s, 0) for s in adjacent_signals}
    interpolated_value = np.mean(adjacent_values.values())
    return interpolated_value


def outlier_z_scores(
                     boc,
                     ac_average,
                     ac_std,
                    ):
    if set(ac_average.keys()) != set(ac_std.keys()):
        raise Exception()
    combined_keys = ac_average.keys() + boc.keys()
    z_scores, undefined_z_scores = {}, {}
    for s, z, si in combined_keys:
        bp = boc.get((s, z, si), 0)
        ap = ac_average.get((s, z, si), 0)
        sp = ac_std.get((s, z, si), 0)
        if sp == 0:
            undefined_z_scores.setdefault((s, z, si), (bp, ap, sp))
        else:
            z_scores.setdefault((s, z, si), float(bp - ap)**2 / float(sp)**2)
    z_scores = {(s, z, si): math.copysign(sqrt(m),
                                          boc.get((s, z, si), 0)
                                          - ac_average.get((s, z, si), 0))
                for (s, z, si), m in z_scores.iteritems()}
    return z_scores, undefined_z_scores


def iterative_peak_finding(boc_raw,
                           boc_percent,
                           ac_average,
                           ac_std,
                           num_cycles,
                           sigma_threshold=3,
                           include_multidrop=False,
                          ):
    peak_list, undefined_peaks = [], []
    updated_boc_raw = dict(boc_raw)
    updated_boc_percent = dict(boc_percent)
    if set(boc_raw.keys()) != set(boc_percent.keys()):
        raise ValueError("boc_raw and boc_percent don't have matching keys.")
    max_iterations = len(updated_boc_percent)
    while max_iterations >= 0:
        max_iterations -= 1
        z_scores, undefined_z_scores = outlier_z_scores(
                                                       boc=updated_boc_percent,
                                                       ac_average=ac_average,
                                                       ac_std=ac_std,
                                                       )
        for (s, z, si), (bp, ap, sp) in undefined_z_scores.iteritems():
            interpolated_percent = interpolate_signal(
                                           signals=updated_boc_raw,
                                           interpolation_target=(s, z, si),
                                           include_multidrop=include_multidrop,
                                           num_cycles=num_cycles,
                                                     )
            updated_boc_raw[(s, z, si)] = interpolated_percent
            updated_boc_percent = counts_to_percent(
                                           updated_boc_raw,
                                           include_remainders=False,
                                           include_multidrop=include_multidrop,
                                           max_cycle=num_cycles + 1,
                                                   )
            undefined_peaks.append((s, z, si, bp, ap, sp))
        if len(z_scores) == 0:
            break
        outlier = max(z_scores, key=z_scores.get)
        if z_scores[outlier] <= sigma_threshold:
            break
        peak_list.append(outlier)
        interpolated_percent = interpolate_signal(
                                           signals=updated_boc_raw,
                                           interpolation_target=outlier,
                                           include_multidrop=include_multidrop,
                                           num_cycles=num_cycles,
                                                 )
        updated_boc_raw[outlier] = interpolated_percent
        updated_boc_percent = counts_to_percent(
                                           updated_boc_raw,
                                           include_remainders=False,
                                           include_multidrop=include_multidrop,
                                           max_cycle=num_cycles + 1,
                                               )
    updated_boc_raw = {(s, z, si): int(round(count))
                       for (s, z, si), count in updated_boc_raw.iteritems()}
    return peak_list, undefined_peaks, updated_boc_raw, updated_boc_percent


def iterative_peak_finding_v2(boc_raw,
                              boc_percent,
                              ac_average,
                              ac_std,
                              num_cycles,
                              sigma_threshold=3,
                              include_multidrop=False,
                             ):
    peak_list, undefined_peaks = [], []
    updated_boc_raw = dict(boc_raw)
    updated_boc_percent = dict(boc_percent)
    if set(boc_raw.keys()) != set(boc_percent.keys()):
        raise ValueError("boc_raw and boc_percent don't have matching keys.")
    max_iterations = len(updated_boc_percent)
    last_outlier = None
    while max_iterations >= 0:
        max_iterations -= 1
        z_scores, undefined_z_scores = outlier_z_scores(
                                                       boc=updated_boc_percent,
                                                       ac_average=ac_average,
                                                       ac_std=ac_std,
                                                       )
        for (s, z, si), (bp, ap, sp) in undefined_z_scores.iteritems():
            interpolated_percent = interpolate_signal(
                                           signals=updated_boc_raw,
                                           interpolation_target=(s, z, si),
                                           include_multidrop=include_multidrop,
                                           num_cycles=num_cycles,
                                                     )
            updated_boc_raw[(s, z, si)] = interpolated_percent
            updated_boc_percent = counts_to_percent(
                                           updated_boc_raw,
                                           include_remainders=False,
                                           include_multidrop=include_multidrop,
                                           max_cycle=num_cycles + 1,
                                                   )
            undefined_peaks.append((s, z, si, bp, ap, sp))
        if len(z_scores) == 0:
            break
        outlier = max(z_scores, key=z_scores.get)

        if outlier == last_outlier:
            if len(z_scores) < 2:
                break
            outlier = sorted(z_scores.items(), key=lambda x:x[1])[-2][0]
        last_outlier = outlier

        print("outlier " + str(outlier) + ": " + str(z_scores[outlier])) #DEBUG
        debug_adj = generate_adjacent_positions(signal=outlier, include_multidrop=False)#DEBUG
        #print("debug_adj = " + str(debug_adj))
        print(str(outlier) + ": " + str(updated_boc_raw[outlier]))
        for adj in debug_adj:
            fs = (tuple([('A', pos) for pos in adj]), True, len(adj))
            print(str(fs) + ": " + str(updated_boc_raw.get(fs, 0)))


        if z_scores[outlier] <= sigma_threshold:
            break
        peak_list.append(outlier)
        interpolated_percent = interpolate_signal(
                                           signals=updated_boc_raw,
                                           interpolation_target=outlier,
                                           include_multidrop=include_multidrop,
                                           num_cycles=num_cycles,
                                                 )
        updated_boc_raw[outlier] = interpolated_percent
        updated_boc_percent = counts_to_percent(
                                           updated_boc_raw,
                                           include_remainders=False,
                                           include_multidrop=include_multidrop,
                                           max_cycle=num_cycles + 1,
                                               )
    updated_boc_raw = {(s, z, si): int(round(count))
                       for (s, z, si), count in updated_boc_raw.iteritems()}
    return peak_list, undefined_peaks, updated_boc_raw, updated_boc_percent


def iterative_peak_finding_v3(boc_raw,
                              boc_percent,
                              ac_average,
                              ac_std,
                              num_cycles,
                              sigma_threshold=3,
                              include_multidrop=False,
                              sigma_subtract=None,
                             ):
    peak_list, undefined_peaks = [], []
    updated_boc_raw = dict(boc_raw)
    updated_boc_percent = dict(boc_percent)
    if set(boc_raw.keys()) != set(boc_percent.keys()):
        raise ValueError("boc_raw and boc_percent don't have matching keys.")
    prior_boc_raw = None
    while True:
        z_scores, undefined_z_scores = outlier_z_scores(
                                                       boc=updated_boc_percent,
                                                       ac_average=ac_average,
                                                       ac_std=ac_std,
                                                       )
        for (s, z, si), (bp, ap, sp) in undefined_z_scores.iteritems():
            interpolated_count = interpolate_signal(
                                           signals=updated_boc_raw,
                                           interpolation_target=(s, z, si),
                                           include_multidrop=include_multidrop,
                                           num_cycles=num_cycles,
                                                   )
            updated_boc_raw[(s, z, si)] = interpolated_count
            undefined_peaks.append((s, z, si, bp, ap, sp))
        updated_boc_percent = counts_to_percent(
                                           updated_boc_raw,
                                           include_remainders=False,
                                           include_multidrop=include_multidrop,
                                           max_cycle=num_cycles,
                                               )
        if len(z_scores) == 0:
            break
        outlier = max(z_scores, key=z_scores.get)
        if z_scores[outlier] <= sigma_threshold:
            break
        interpolated_counts = {k:
                               interpolate_signal(
                                           signals=updated_boc_raw,
                                           interpolation_target=k,
                                           include_multidrop=include_multidrop,
                                           num_cycles=num_cycles,
                                                 )
                               for k, z in z_scores.iteritems()}
        z_diffs = {}
        for k, interpolated_count in interpolated_counts.iteritems():
            if z_scores[k] <= sigma_threshold:
                continue
            temp_updated_raw = dict(updated_boc_raw)
            temp_updated_raw[k] = interpolated_count
            temp_updated_boc_percent = counts_to_percent(
                                           temp_updated_raw,
                                           include_remainders=False,
                                           include_multidrop=include_multidrop,
                                           max_cycle=num_cycles,
                                                        )
            temp_z_scores, temp_undefined_z_scores = outlier_z_scores(
                                                  boc=temp_updated_boc_percent,
                                                  ac_average=ac_average,
                                                  ac_std=ac_std,
                                                                     )
            z_diff = z_scores[k] - temp_z_scores[k]
            z_diffs.setdefault(k, z_diff)
        best_improvement = max(z_diffs, key=z_diffs.get)
        if z_diffs[best_improvement] <= 0:
            break
        outlier = best_improvement
        updated_boc_raw[outlier] = interpolated_counts[outlier]
        if prior_boc_raw is not None:
            assert set(prior_boc_raw.keys()) == set(updated_boc_raw.keys())
            abs_diffs = [abs(updated_boc_raw[k] - prior_boc_raw[k])
                         for k, v in prior_boc_raw.iteritems()]
            if max(abs_diffs) < 0.001:
                break
        prior_boc_raw = dict(updated_boc_raw)
        updated_boc_percent = counts_to_percent(
                                           updated_boc_raw,
                                           include_remainders=False,
                                           include_multidrop=include_multidrop,
                                           max_cycle=num_cycles,
                                               )
    updated_boc_raw = {(s, z, si): int(round(count))
                       for (s, z, si), count in updated_boc_raw.iteritems()}
    if sigma_subtract is not None:
        if set(ac_average.keys()) != set(ac_std.keys()):
            raise ValueError("ac_average and ac_std keys don't match. "
                             + "set(ac_average.keys()) ^ set(ac_std.keys()) = "
                             + str(set(ac_average.keys()) ^ 
                                   set(ac_std.keys())))
        assert set(updated_boc_percent.keys()) == set(updated_boc_raw.keys())
        for (s, z, si), percent in updated_boc_percent.items():
            if percent == 0:
                continue
            plus_sigma_ratio = (float(percent + ac_std.get((s, z, si), 0))
                                / percent)
            updated_boc_raw[(s, z, si)] = \
                     int(round(updated_boc_raw[(s, z, si)] * plus_sigma_ratio))
        updated_boc_percent = counts_to_percent(
                                           updated_boc_raw,
                                           include_remainders=False,
                                           include_multidrop=include_multidrop,
                                           max_cycle=num_cycles,
                                               )
    return peak_list, undefined_peaks, updated_boc_raw, updated_boc_percent


def subtract_false_positives(background_boc_raw,
                             background_boc_percent,
                             counts_above_background,
                             ac_std,
                             expected_false_positive_percent=5.0,
                            ):
    """
    Diminish counts_above_background until expected number of false positives
    due to variation in background rate is at or below
    expected_false_positive_percent.

    We assume standard deviation around the corrected background is identical
    to that of collective acetylated data.
    """
    if not (set(background_boc_raw.keys())
            == set(background_boc_percent.keys())
            == set(counts_above_background.keys())
           ):
        debug_output = "\n\n"
        debug_output += ("sorted(background_boc_raw.keys()) =\n"
                         + str(sorted(background_boc_raw.keys()))
                         + "\nsorted(background_boc_percent.keys()) =\n"
                         + str(sorted(background_boc_percent.keys()))
                         + "\nsorted(counts_above_background.keys()) =\n"
                         + str(sorted(counts_above_background.keys()))
                        )
        debug_output += "\n\n"
        debug_output += ("set(background_boc_raw.keys()) ^ "
                         + "set(background_boc_percent.keys()) = "
                         + str(set(background_boc_raw.keys())
                               ^ set(background_boc_percent.keys()))
                         + "\nset(background_boc_raw.keys()) ^ "
                         + "set(counts_above_background.keys()) = "
                         + str(set(background_boc_raw.keys())
                               ^ set(counts_above_background.keys()))
                         + "\nset(background_boc_percent.keys()) ^ "
                         + "set(counts_above_background.keys()) = "
                         + str(set(background_boc_percent.keys())
                               ^ set(counts_above_background.keys()))
                        )
        raise ValueError("Keys for all three dictionaries -- "
                         + "background_boc_raw, background_boc_percent, "
                         + "counts_above_background -- must match."
                         + debug_output
                        )
    if (False #For now, treat ac_std[(s, z, si)] == 0 as undefined
        and not set(counts_above_background.keys()) <= set(ac_std.keys())):
        set_diff = set(counts_above_background.keys()) - set(ac_std.keys())
        debug_output = "".join([
                             ("\n" + str((s, z, si)) + " "
                              + str(counts_above_background.get((s, z, si), 0))
                              + " " + str(ac_std.get((s, z, si), 0))
                             )
                                for (s, z, si) in sorted(tuple(set_diff))
                               ])
        raise ValueError("ac_std not defined for all keys in "
                         + "counts_above_background: "
                         + debug_output
                        )
    #Determine sigma in terms of absolute counts.
    #These can be used to make z-score type assessments.
    sigma_counts, undefined_sigma = {}, {}
    for (s, z, si), count in background_boc_raw.iteritems():
        if count == 0:
            if background_boc_percent[(s, z, si)] > 0.0001:
                raise Exception("count is 0, but background_boc_percent["
                                + str((s, z, si)) + "] is not approx zero, "
                                + "instead it is "
                                + str(background_boc_percent[(s, z, si)]))
            continue #this doesn't even make it into consideration
        elif background_boc_percent[(s, z, si)] == 0:
            raise Exception("background_boc_percent[" + str((s, z, si)) + "] "
                            + "is zero, but count is positive " + str(count))
        elif background_boc_percent[(s, z, si)] < 0:
            raise Exception("background_boc_percent cannot be negative: "
                            + "background_boc_percent[" + str((s, z, si))
                            + "] = " + str(background_boc_percent[(s, z, si)]))
        if (s, z, si) not in ac_std or ac_std[(s, z, si)] == 0:
            undefined_sigma.setdefault((s, z, si),
                                       background_boc_percent[(s, z, si)])
            continue
        std_ratio = (float(ac_std[(s, z, si)])
                     / background_boc_percent[(s, z, si)])
        sigma_counts.setdefault((s, z, si),
                                std_ratio * background_boc_raw[(s, z, si)])
    def fp_count(#background_count, #b
                 count_above_background, #n
                 subtract_count, #T(hreshold)
                 sigma, #in absolute count number
                ):
        expected_value_sum = 0.0
        normal_approximation = norm(loc=0, scale=sigma)
        assert subtract_count >= 0
        for t in range(subtract_count + 1, count_above_background + 1):
            z_score_probability = normal_approximation.pdf(t - 0.5)
            expected_value_sum += (t - subtract_count) * z_score_probability
        return expected_value_sum
    subtractions = {}
    for (s, z, si), sigma in sigma_counts.iteritems():
        if counts_above_background[(s, z, si)] == 0:
            continue
        subtract = counts_above_background[(s, z, si)]
        for T in range(counts_above_background[(s, z, si)]):
            fpc = fp_count(
                    count_above_background=counts_above_background[(s, z, si)],
                    subtract_count=T,
                    sigma=sigma,
                          )
            fp_percent = (float(fpc)
                          / (counts_above_background[(s, z, si)] - T)
                          * 100.0)
            if fp_percent <= expected_false_positive_percent:
                subtract = T
                break
        subtractions.setdefault((s, z, si), subtract)
    return subtractions, undefined_sigma, sigma_counts


def expected_background(background_boc_raw,
                        background_boc_percent,
                        ac_std,
                       ):
    if set(background_boc_raw.keys()) != set(background_boc_percent.keys()):
        debug_output = "\n\n"
        debug_output += ("sorted(background_boc_raw.keys()) =\n"
                         + str(sorted(background_boc_raw.keys()))
                         + "\nsorted(background_boc_percent.keys()) =\n"
                         + str(sorted(background_boc_percent.keys()))
                        )
        debug_output += "\n\n"
        debug_output += ("set(background_boc_raw.keys()) ^ "
                         + "set(background_boc_percent.keys()) = "
                         + str(set(background_boc_raw.keys())
                               ^ set(background_boc_percent.keys()))
                        )
        raise ValueError("Keys for background_boc_raw and "
                         + "background_boc_percent must match."
                         + debug_output
                        )
    sigma_counts, undefined_sigma = {}, {}
    for (s, z, si), count in background_boc_raw.iteritems():
        if count == 0:
            if background_boc_percent[(s, z, si)] > 0.0001:
                raise Exception("count is 0, but background_boc_percent["
                                + str((s, z, si)) + "] is not approx zero, "
                                + "instead it is "
                                + str(background_boc_percent[(s, z, si)]))
            continue #this doesn't even make it into consideration
        elif background_boc_percent[(s, z, si)] == 0:
            raise Exception("background_boc_percent[" + str((s, z, si)) + "] "
                            + "is zero, but count is positive " + str(count))
        elif background_boc_percent[(s, z, si)] < 0:
            raise Exception("background_boc_percent cannot be negative: "
                            + "background_boc_percent[" + str((s, z, si))
                            + "] = " + str(background_boc_percent[(s, z, si)]))
        if (s, z, si) not in ac_std or ac_std[(s, z, si)] == 0:
            undefined_sigma.setdefault((s, z, si),
                                       background_boc_percent[(s, z, si)])
            continue
        std_ratio = (float(ac_std[(s, z, si)])
                     / background_boc_percent[(s, z, si)])
        sigma_counts.setdefault((s, z, si),
                                std_ratio * background_boc_raw[(s, z, si)])
    expected_counts = {}
    for (s, z, si), sigma in sigma_counts.iteritems():
        normal_approximation = norm(loc=0, scale=sigma)
        expected_value_sum = 0.0
        for t in range(int(ceil(sigma * 7.0))):
            z_score_probability = normal_approximation.pdf(t - 0.5)
            expected_value_sum += z_score_probability * t
        expected_counts.setdefault((s, z, si), int(round(expected_value_sum)))
    return expected_counts
