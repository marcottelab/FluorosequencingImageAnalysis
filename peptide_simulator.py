from collections import namedtuple, deque, defaultdict
import random
import multiprocessing
from string import letters, digits
from math import log, floor, e
import numpy as np
from itertools import combinations
from MCsimlib import _pairwise


FluorEvent = namedtuple('FluorEvent',
                        [
                         'original_position',
                         'original_amino_acid',
                         'event_name',
                         'cycle_number',
                         'message',
                        ]
                       )


def _define_reserved_character(sequence, labels):
        sequence_characters = set([L for L in sequence])
        label_characters = set([L for L in labels])
        characters_used = label_characters | sequence_characters
        possible_characters = (set([L for L in letters])
                               | set([d for d in digits]))
        characters_available = possible_characters - characters_used
        if len(characters_available) == 0:
            raise ValueError("sequence and labels use all possible "
                             "string.letters and string.digits. At least one "
                             "must remain available as a reserved letter for "
                             "simulation purposes.")
        return characters_available.pop()


def _make_mock(reserved_character, labels, success_event_name=None,
               failure_event_name=None, **experimental_parameters):
    def _mock(molecule, event_buffer, cycle_number):
        pass
    return _mock


def _make_edman(reserved_character, labels, success_event_name='edman',
                failure_event_name='edman failure', **experimental_parameters):
    p = experimental_parameters['p']
    def _edman(molecule, event_buffer, cycle_number):
        if len(molecule) > 0:
            nterm_position, nterm_amino_acid = molecule[0]
            random_point = random.random()
            if random_point < p:
                if nterm_amino_acid in labels:
                    emission = FluorEvent(
                                          original_position=nterm_position,
                                          original_amino_acid=nterm_amino_acid,
                                          event_name=success_event_name,
                                          cycle_number=cycle_number[0],
                                          message=None,
                                         )
                    event_buffer.append(emission)
                    molecule.pop(0)
                else:
                    molecule.pop(0)
            else:
                error = FluorEvent(
                                   original_position=nterm_position,
                                   original_amino_acid=nterm_amino_acid,
                                   event_name=failure_event_name,
                                   cycle_number=cycle_number[0],
                                   message=None,
                                  )
                event_buffer.append(error)
        else:
            pass
    return _edman


def _make_tirf(reserved_character, labels, success_event_name=None,
               failure_event_name='dye destruction',
               **experimental_parameters):
    """Photobleaching events are assumed to occur during an exposure."""
    per_cycle_b = experimental_parameters.get('per_cycle_b',
                                              e**-experimental_parameters['b'])
    def _tirf(molecule, event_buffer, cycle_number):
        for i, (position, amino_acid) in enumerate(molecule):
            if amino_acid not in labels:
                continue
            random_point = random.random()
            if random_point > per_cycle_b:
                emission = FluorEvent(
                                      original_position=position,
                                      original_amino_acid=amino_acid,
                                      event_name=failure_event_name,
                                      cycle_number=cycle_number[0],
                                      message=None,
                                     )
                event_buffer.append(emission)
                molecule[i] = (reserved_character, position)
    return _tirf


def _make_dud(reserved_character, labels, success_event_name=None,
              failure_event_name='dye dud', **experimental_parameters):
    u = experimental_parameters['u']
    def _dud(molecule, event_buffer, cycle_number):
        for i, (position, amino_acid) in enumerate(molecule):
            if amino_acid not in labels:
                continue
            random_point = random.random()
            if random_point < u:
                error = FluorEvent(
                                   original_position=position,
                                   original_amino_acid=amino_acid,
                                   event_name=failure_event_name,
                                   cycle_number=cycle_number[0],
                                   message=None,
                                  )
                event_buffer.append(error)
                molecule[i] = (reserved_character, position)
    return _dud


def _increment_cycle(molecule, event_buffer, cycle_number):
    """cycle_number = [int]"""
    cycle_number[0] = cycle_number[0] + 1


def _make_count_dyes(reserved_character, labels,
                     success_event_name='dye count', failure_event_name=None,
                     **experimental_parameters):
    def _count_dyes(molecule, event_buffer, cycle_number):
        #no defaultdict(int) because needs to count even those not present
        fluor_counts = {L: 0 for L in labels}
        for position, amino_acid in molecule:
            if amino_acid in labels:
                fluor_counts[amino_acid] += 1
        fluor_count = FluorEvent(
                                 original_position=None,
                                 original_amino_acid=None,
                                 event_name=success_event_name,
                                 cycle_number=cycle_number[0],
                                 message=fluor_counts,
                                )
        event_buffer.append(fluor_count)
    return _count_dyes


def _make_strip_surface(reserved_character, labels, success_event_name=None,
                        failure_event_name='surface strip',
                        **experimental_parameters):
    s, sc = experimental_parameters['s'], experimental_parameters['sc']
    s2 = experimental_parameters['s2']
    def _strip_surface(molecule, event_buffer, cycle_number):
        using_s = s if cycle_number[0] <= sc else s2
        random_point = random.random()
        if random_point < using_s:
            for i, (position, amino_acid) in enumerate(molecule):
                if amino_acid not in labels:
                    continue
                stripping = FluorEvent(
                                       original_position=position,
                                       original_amino_acid=amino_acid,
                                       event_name=failure_event_name,
                                       cycle_number=cycle_number[0],
                                       message=None,
                                      )
                event_buffer.append(stripping)
                molecule[i] = (reserved_character, position)
    return _strip_surface


def _make_get_dye_positions(reserved_character, labels,
                            success_event_name='dye count',
                            failure_event_name=None, **experimental_parameters):
    def _get_dye_positions(molecule, event_buffer, cycle_number):
        positions = tuple([(position, amino_acid)
                           for position, amino_acid in molecule
                           if amino_acid in labels])
        dye_positions = FluorEvent(
                                   original_position=None,
                                   original_amino_acid=None,
                                   event_name=success_event_name,
                                   cycle_number=cycle_number[0],
                                   message=positions,
                                  )
        event_buffer.append(dye_positions)
    return _get_dye_positions


def simulate_dye_counts(sequence, labels, num_mocks, num_edmans,
                        num_simulations=1, random_seed=None,
                        reserved_character=None, **experimental_parameters):
    """Assumes C-term attachment."""
    if random_seed is not None:
        random.seed(random_seed)
    else:
        random.seed()
    if reserved_character is None:
        reserved_character = _define_reserved_character(sequence=sequence,
                                                        labels=labels)
    labels = set([L for L in labels])
    _dud = _make_dud(
                     reserved_character=reserved_character,
                     labels=labels,
                     success_event_name=None,
                     failure_event_name='dye dud',
                     **experimental_parameters
                    )
    _mock = _make_mock(
                       reserved_character=reserved_character,
                       labels=labels,
                       success_event_name=None,
                       failure_event_name=None,
                       **experimental_parameters
                      )
    _edman = _make_edman(
                         reserved_character=reserved_character,
                         labels=labels,
                         success_event_name='edman',
                         failure_event_name='edman failure',
                         **experimental_parameters
                        )
    _tirf = _make_tirf(
                       reserved_character=reserved_character,
                       labels=labels,
                       success_event_name=None,
                       failure_event_name='dye destruction',
                       **experimental_parameters
                      )
    _count_dyes = _make_count_dyes(
                                   reserved_character=reserved_character,
                                   labels=labels,
                                   success_event_name='dye count',
                                   failure_event_name=None,
                                   **experimental_parameters
                                  )
    _strip_surface = _make_strip_surface(
                                         reserved_character=reserved_character,
                                         labels=labels,
                                         success_event_name=None,
                                         failure_event_name='surface strip',
                                         **experimental_parameters
                                        )
    _get_dye_positions = _make_get_dye_positions(
                                         reserved_character=reserved_character,
                                         labels=labels,
                                         success_event_name=None,
                                         failure_event_name='dye positions',
                                         **experimental_parameters
                                                )
    experimental_sequence = (
                             [
                              _dud,
                              _tirf,
                              _count_dyes,
                              _get_dye_positions,
                              _increment_cycle,
                             ]

                             + [
                                _mock,
                                _strip_surface,
                                _tirf,
                                _count_dyes,
                                _get_dye_positions,
                                _increment_cycle,
                               ] * num_mocks

                             + [
                                _edman,
                                _strip_surface,
                                _tirf,
                                _count_dyes,
                                _get_dye_positions,
                                _increment_cycle,
                               ] * num_edmans
                            )
    results = []
    num_cycles = num_mocks + num_edmans
    for sim_number in range(num_simulations):
        molecule = list(enumerate(sequence, start=1))
        event_buffer = []
        cycle_number = [0]
        for action in experimental_sequence:
            action(
                   molecule=molecule,
                   event_buffer=event_buffer,
                   cycle_number=cycle_number,
                  )
        dye_decrements = []
        dye_counts = defaultdict(list)
        dye_position_tracker = []
        for event in event_buffer:
            if (event.event_name == 'edman'
                or event.event_name == 'dye destruction'
                or event.event_name == 'dye dud'
                or event.event_name == 'surface strip'):
                dye_decrements.append((event.original_amino_acid,
                                       event.cycle_number))
            elif event.event_name == 'dye count':
                dye_count = event.message
                for label, count in dye_count.iteritems():
                    dye_counts[label].append(count)
            elif event.event_name == 'dye positions':
                dye_positions = event.message
                dye_position_tracker.append(dye_positions)
            else:
                pass
        dye_counts = {label: tuple(count)
                      for label, count in dye_counts.iteritems()}
        dye_decrements = tuple(sorted(dye_decrements, key=lambda x:x[1]))
        dye_position_tracker = tuple(dye_position_tracker)
        results.append((
                        dye_decrements,
                        dye_counts,
                        event_buffer,
                        dye_position_tracker,
                      ))
    return results


def simulate_photometries(dye_counts, beta, beta_sigma, number, ddif=None,
                          dye_position_tracker=None, distance_ddif=None,
                          superdye_rate=0, superdye_factor=1):
    """
    distance_ddif = {distance: ddif}, default = 0; additive.

    superdye_rate: Chance for dye to be a superdye, e.g. superdye_rate=0.5
        means any individual dye has a 50% chance of being a superdye.
    superdye_factor: Superdyes are brighter than normal by
        superdye_coefficient.
    """
    category = tuple([False if seq == 0 else True for seq in dye_counts])
    if not (0 <= superdye_rate <= 1):
        raise ValueError("superdye_rate must be between 0 and 1 (inclusive).")
    num_starting_dyes, num_remaining_dyes = dye_counts[0], dye_counts[-1]
    dye_drops = [0] + [dye_counts[i] - c for i, c in enumerate(dye_counts[1:])]
    assert sum(dye_drops) == num_starting_dyes - num_remaining_dyes
    all_superdye_increments = []
    for n in range(number):
        superdye_increments = [0] * len(dye_drops)
        for d, drop_size in enumerate(dye_drops):
            for i in range(drop_size):
                if random.random() < superdye_rate:
                    superdye_increments[d] += 1
        remainder_superdyes = [1 for d in range(num_remaining_dyes)
                               if random.random() < superdye_rate]
        num_remainder_superdyes = sum(remainder_superdyes)
        superdye_increments[-1] += num_remainder_superdyes
        superdye_increments = [sum(superdye_increments[i:])
                               for i, increment
                               in enumerate(superdye_increments)]
        all_superdye_increments.append(superdye_increments)
    if superdye_rate == 0:
        assert np.sum(np.array(all_superdye_increments)) == 0
    if distance_ddif is not None:
        if dye_position_tracker is None:
            raise ValueError("distance_ddif requires dye_position_tracker.")
        intensities = []
        for dye_positions in dye_position_tracker:
            num_dyes = len(dye_positions)
            if num_dyes == 0:
                intensities.append([0.0] * number)
                continue
            dye_position_pairs = combinations(dye_positions, 2)
            dye_distance_lists = defaultdict(list)
            for (pos1, aa1), (pos2, aa2) in dye_position_pairs:
                distance = abs(pos2 - pos1)
                dye_distance_lists[pos1].append(distance)
                dye_distance_lists[pos2].append(distance)
            per_dye_ddif = [sum([distnace_ddif.get(distance, 0)
                                 for distance in dye_distance_lists[position]])
                            for position, amino_acid in dye_positions]
            total_ddif = sum(per_dye_ddif)
            if superdye_rate == 0:
                intensities.append(
                               np.random.lognormal(
                                   mean=log(beta) + log(num_dyes) - total_ddif,
                                   sigma=beta_sigma,
                                   size=number,
                                                  )
                                  )
            else:
                intensities.append([])
                for n in range(number):
                    superdye_increment = \
                                   all_superdye_increments[n][len(intensities)]
                    mean = (
                            log(beta)
                            + log(num_dyes + superdye_increment * superdye_factor)
                            - total_ddif
                           )
                    intensities[-1].append(float(
                                          np.random.lognormal(
                                                              mean=mean,
                                                              sigma=beta_sigma,
                                                              size=1,
                                                             )[0]
                                                )
                                          )
    else:
        if ddif is None:
            ddif = [0.0] * len(dye_counts)
        intensities = []
        if superdye_rate == 0:
            intensities = [np.random.lognormal(
                                     mean=log(beta) + log(seq) - ddif[seq - 1],
                                     sigma=beta_sigma,
                                     size=number,
                                              )
                           if seq > 0
                           else [0.0] * number
                           for seq in dye_counts]
        else:
            for s, seq in enumerate(dye_counts):
                if seq == 0:
                    intensities.append([0.0] * number)
                    continue
                intensities.append([])
                for n in range(number):
                    superdye_increment = all_superdye_increments[n][s]
                    mean = (
                            log(beta)
                            + log(seq + superdye_increment * superdye_factor)
                            - ddif[seq - 1]
                           )
                    intensities[-1].append(float(
                                          np.random.lognormal(
                                                              mean=mean,
                                                              sigma=beta_sigma,
                                                              size=1,
                                                             )[0]
                                                )
                                          )
    return category, tuple(zip(*intensities))


def peptide_simulation(sequence, labels, num_mocks, num_edmans,
                       num_simulations=1, random_seed=None, num_processes=None,
                       reserved_character=None, **experimental_parameters):
    """Assumes C-term attachment."""
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    labels = set([L for L in labels])
    sims_per_process = int(floor(num_simulations / num_processes))
    remainder_sim_count = num_simulations % num_processes
    process_sim_loads = [sims_per_process for p in range(num_processes)]
    for i, load in enumerate(process_sim_loads[:remainder_sim_count]):
        process_sim_loads[i] = load + 1
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=None)
    processes = deque()
    for i, load in enumerate(process_sim_loads):
        process = pool.apply_async(simulate_dye_counts,
                                   (
                                    sequence,
                                    labels,
                                    num_mocks,
                                    num_edmans,
                                    load,
                                    random.random(),
                                    reserved_character,
                                   ),
                                    experimental_parameters,
                                  )
        processes.append(process)
    pool.close()
    pool.join()
    merged_dye_count_results = deque()
    beta = experimental_parameters['beta']
    beta_sigma = experimental_parameters['beta_sigma']
    ddif = experimental_parameters.get('ddif', None)
    distance_ddif = experimental_parameters.get('distance_ddif', None)
    superdye_rate = experimental_parameters.get('superdye_rate', 0)
    superdye_factor = experimental_parameters.get('superdye_factor', 2)
    while processes:
        process = processes.pop()
        results = process.get()
        while results:
            (
             dye_decrements,
             dye_counts,
             event_buffer,
             dye_position_tracker,
            ) = results.pop()
            categories_and_intensities = {L: simulate_photometries(
                                     dye_counts=counts,
                                     beta=beta,
                                     beta_sigma=beta_sigma,
                                     number=1,
                                     ddif=ddif,
                                     dye_position_tracker=dye_position_tracker,
                                     distance_ddif=distance_ddif,
                                     superdye_rate=superdye_rate,
                                     superdye_factor=superdye_factor,
                                                                  )
                                          for L, counts
                                          in dye_counts.iteritems()}
            merged_dye_count_results.append((dye_decrements,
                                             dye_counts,
                                             event_buffer,
                                             categories_and_intensities))
    return merged_dye_count_results


def convert_to_oldstyle(merged_dye_count_results):
    """
    Convert peptide_simulator.peptide_simulation output to prior format.

    Before peptide_simulator, (('A', 0),) represented no signal/dye-decrement.
    In peptide_simulator, however, it represents the removal of a fluor before
    any experimental cycles (e.g. duds).

    This function converts merged_dye_count_results as returned by
    peptide_simulation into the old definition. Molecules that start with a 0
    dye count are omitted to stay consistent with the oldstyle definition. They
    were not included before peptide_simulator anyways.

    Furthermore, the prior format only dealt with one label at a time, which
    was always indicated using the amino acid code 'A'. This function converts
    only results with one amino acid, and converts it to 'A'.

    event_buffer, however was not defined in the prior format, and remains
    unchanged.
    """
    oldstyle_results = deque()
    for (dye_decrements, dye_counts,
         event_buffer, categories_and_intensities) in merged_dye_count_results:
        amino_acid_set = set([amino_acid
                              for amino_acid, position in dye_decrements])
        if len(amino_acid_set) > 1:
            raise Exception("Oldstyle only works with one label.")
        oldstyle_dye_decrements = tuple([('A', position)
                                         for amino_acid, position
                                         in dye_decrements
                                         if position != 0])
        if len(dye_counts) > 1:
            raise Exception("Oldstyle only works with one label.")
        dye_counts_drops = sum([c1 - c2
                                for c1, c2
                                in _pairwise(next(dye_counts.itervalues()))])
        if len(oldstyle_dye_decrements) == 0:
            oldstyle_dye_decrements = (('A', 0),)
            assert dye_counts_drops == 0, \
                             ("dye_counts_drops != 0 "
                               + "when oldstyle_dye_decrements is (('A', 0),)")
        else:
            assert dye_counts_drops == len(oldstyle_dye_decrements), \
                (str(dye_counts_drops) + " != "
                 + str(len(oldstyle_dye_decrements))
                 + "; must be dye_counts_drops == "
                 + "len(oldstyle_dye_decrements); dye_counts = "
                 + str(dye_counts) + ", dye_decrements = "
                 + str(dye_decrements) + ", oldstyle_dye_decrements = "
                 + str(oldstyle_dye_decrements))
        oldstyle_categories_and_intensities = \
                                    {'A': (category, (intensities,))
                                     for label, (category, (intensities,))
                                     in categories_and_intensities.iteritems()
                                     if True in category}
        if oldstyle_categories_and_intensities:
            oldstyle_results.append((oldstyle_dye_decrements,
                                     dye_counts,
                                     event_buffer,
                                     oldstyle_categories_and_intensities))
        else:
            #omits results where all labels are 0 before the first cycle
            pass
    return oldstyle_results
