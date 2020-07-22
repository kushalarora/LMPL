from typing import List

from quant_exp_bias.oracles.oracle_base import Oracle
from nltk import PCFG
from nltk.grammar import Nonterminal
from nltk.parse.pchart import InsideChartParser

from scipy.stats import zipf

from functools import reduce
import logging
import itertools
import random
import re
import subprocess
import time
import string
import numpy as np

import time

from multiprocessing import Pool
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Init routine for pool so that these objects aren't instantiated evertime.
# https://stackoverflow.com/questions/9944370/use-of-initialize-in-python-multiprocessing-worker-pool
parser = None
grammar = None


def init_pool(grammar_string):
    global parser
    global grammar
    grammar = PCFG.fromstring(grammar_string)
    parser = InsideChartParser(grammar)
    return parser


@Oracle.register('artificial_lang_oracle')
class ArtificialLanguageOracle(Oracle):
    """
    TODO (Kushal): Expand class doc.
    SO: https://stackoverflow.com/questions/15009656/how-to-use-nltk-to-generate-sentences-from-an-induced-grammar
    """

    def __init__(self,
                 grammar_file: str,
                 use_weighted_choice: bool = True,
                 parallelize=True,
                 num_threads=64,
                 max_len=50,
                 min_len=0):
        """ TODO (Kushal): Add function doc.
        """
        super(Oracle, self).__init__()

        self._grammar_string = open(grammar_file).read()
        self._use_weighted_choice = use_weighted_choice

        self._parallelize = parallelize

        self._num_threads = num_threads

        self._max_len = max_len

        self._min_len = min_len

        self._pool = Pool(self._num_threads, init_pool, [self._grammar_string])

    @staticmethod
    def generate_grammar_string(grammar_template_file: str,
                                vocabulary_size: int,
                                vocabulary_distribution: str,
                                epsilon=0,
                                ):

        # TODO (Kushal): Even with smoothing it will fail to
        # parse SOS a b EOS it will not reach end state. So,
        # We should add an end state at each state transition.
        def _get_vocab_prob(vsize, offset=0):
            if vocabulary_distribution == 'zipf':
                dist = zipf(1.2)
                p_vocab = [dist.pmf(x + 1) for x in range(vsize)]
                p_vocab /= sum(p_vocab)
            elif vocabulary_distribution == 'uniform':
                p_vocab = [1.0/vsize] * vsize

            p_vocab = [(1.0 - offset) * x for x in p_vocab]

            return p_vocab

        printables = [f"'{x}'" for x in string.ascii_letters if x not in set(["'", '"'])]
        assert vocabulary_size <= len(printables)
        vocab = [printables[i] for i in range(vocabulary_size)]
        extended_vocab = ["'SOS'", "'EOS'"] + vocab

        grammar_template = open(grammar_template_file)
        grammar_rules = []

        groupidx2name = {} 
        for template in grammar_template:
            states_and_inputs = template.strip().split()
            current_state, arrow, next_states = states_and_inputs[0], states_and_inputs[1], states_and_inputs[2:]
            if len(next_states) == 2:
                inp, next_state = next_states
            elif len(next_states) == 3:
                inp, next_state, prob = next_states

            match = re.match("'<G([0-9])+>'", inp)
            if match:
                group_id = int(match.groups()[0])
                groupidx2name[group_id] = inp

            elif inp not in extended_vocab:
                extended_vocab.append(inp)

        group2idx = {}
        for i, g in groupidx2name.items():
            group2idx[g] = i

        num_groups = len(groupidx2name)
        group_vocab_size = vocabulary_size//num_groups

        current_state_offset = {}
        grammar_template.seek(0)
        for template in grammar_template:
            token2p = {}
            states_and_inputs = template.strip().split()
            current_state, arrow, next_states = states_and_inputs[0], states_and_inputs[1], states_and_inputs[2:]

            if current_state not in current_state_offset:
                current_state_offset[current_state] = 0

            offset = current_state_offset[current_state]
            prob = None
            if len(next_states) == 2:
                inp, next_state = next_states
            elif len(next_states) == 3:
                inp, next_state, prob = next_states
                prob = float(prob)

            if inp == "'EOS'":
                token2p[inp] = (prob or 1.0 - offset) - epsilon * (vocabulary_size + 2)

                for token in extended_vocab:
                    if token in token2p:
                        p = token2p[token]
                    else:
                        p = epsilon

                    current_state_offset[current_state] += p
                    grammar_rules.append(f"{current_state} {arrow} {token} [{p:.5f}]")
            else:
                if re.match("'<G[0-9]+>'", inp):
                    group_num = group2idx[inp]
                    group_vocab = vocab[group_num * group_vocab_size: (group_num + 1) * group_vocab_size]
                    group_p_vocab = _get_vocab_prob(len(group_vocab), offset)
                    if prob:
                        group_p_vocab = [prob * x for x in group_p_vocab]

                    for token, p in zip(group_vocab, group_p_vocab):
                        token2p[token] = p - epsilon * (vocabulary_size + 2)/len(group_p_vocab)
                else:
                    token2p[inp] = (prob or 1.0 - offset) - epsilon * (vocabulary_size + 2)

                for token in extended_vocab:
                    if token in token2p:
                        p = token2p[token]
                    else:
                        p = epsilon
                        # p = epsilon/2
                        # grammar_rules.append(f"{current_state} {arrow} {token} [{epsilon/2:.5f}]")

                    current_state_offset[current_state] += p
                    grammar_rules.append(f"{current_state} {arrow} {token} {next_state} [{p:.5f}]")

        grammar_string = ""
        for rule in grammar_rules:
            grammar_string += f"{rule}\n"
        return grammar_string

    @staticmethod
    def _weighted_choice(productions):
        """ TODO (Kushal): Add function doc.
        """
        prods_with_probs = [(prod, prod.prob()) for prod in productions]
        total = sum(prob for prod, prob in prods_with_probs)
        r = random.uniform(0, total)
        upto = 0
        for prod, prob in prods_with_probs:
            if upto + prob >= r:
                return prod
            upto += prob
        assert False, "Shouldn't get here"

    @staticmethod
    def _rewrite_at(index, replacements, the_list):
        """ TODO (Kushal): Add function doc.
        """
        del the_list[index]
        the_list[index:index] = replacements

    @staticmethod
    def generate_sequence(grammar_string, use_weighted_choice):
        """ TODO (Kushal): Add function doc.
        """
        global grammar
        sentence_list = [grammar.start()]
        all_terminals = False
        choice = ArtificialLanguageOracle._weighted_choice if use_weighted_choice else random.choice
        while not all_terminals:
            all_terminals = True
            for position, symbol in enumerate(sentence_list):
                if symbol in grammar._lhs_index:
                    all_terminals = False
                    derivations = grammar._lhs_index[symbol]
                    derivation = choice(derivations)
                    ArtificialLanguageOracle._rewrite_at(position,
                                                         derivation.rhs(),
                                                         sentence_list)
        return ' '.join(sentence_list)

    def sample_training_set(self, num_samples: int):
        """ TODO (Kushal): Add function doc.
        """
        # TODO (Kushal): Reformat the code to move generator to the base class and derived class only overloads generate_sequence method.
        # with Pool(self._num_threads, init_pool, [self._grammar_string]) as pool:
        #     samples = pool.starmap(ArtificialLanguageOracle.generate_sequence, [(self._grammar_string, self._use_weighted_choice)]* num_samples * 2)
        # for sample in samples:
        #     if (len(sample) <= self._max_len) and (len(sample) >= self._min_len):
        #         outputs.append(sample)
        # return outputs[:num_samples]

        outputs = set([])
        while len(outputs) < num_samples:
            samples = self._pool.starmap(ArtificialLanguageOracle.generate_sequence,
                                         [(self._grammar_string, self._use_weighted_choice)] * num_samples * 2)
            for sample in samples:
                if (len(sample) <= self._max_len) and (len(sample) >= self._min_len):
                    outputs.add(sample)
        return list(outputs)[:num_samples]

    def compute_sent_probs(self, sequences: List[List[str]]):
        """ TODO (Kushal): Add function doc.
        """
        # TODO (Kushal): Reformat the code to move the for loop in the base class.
        # with Pool(self._num_threads, init_pool, [self._grammar_string]) as pool:
        # return self._pool.starmap(ArtificialLanguageOracle._compute_one_sent_prob, [(sequence, ) for sequence in sequences])
        return self._pool.starmap(ArtificialLanguageOracle._compute_one_sent_prob, [(sequence, ) for sequence in sequences])

    @staticmethod
    def _compute_one_sent_prob(sequence: List[str]):
        global parser
        probs = 1e-100 * len(sequence)
        cond_probs = []
        try:
            parses = list(parser.parse(sequence))
            if parses and len(parses) > 0:
                    # We will only consider top parse as
                    # other are because of smoothing.
                parse = parses[0]

                # Marginalizing by seq_len + 1 because we assume it emits </S> symbol at the end with prob. 1.
                probs = np.exp(np.log(parse.prob() + 1e-100)/(len(sequence) + 1))

                st_probs = [st.prob() for st in parse.subtrees()]
                for i in range(len(st_probs) - 1):
                    cond_probs.append(st_probs[i]/st_probs[i+1])
                cond_probs.append(st_probs[-1])
                cond_probs.append(1.0)
        except Exception as e:
            # Ideally if you fail to parse, the prob is zero.
            # logging.warn(e)
            pass

        if len(cond_probs) == 0:
            cond_probs = [1e-100] * (len(sequence) + 1)
        return probs, cond_probs

    def __del__(self):
        self._pool.terminate()
        time.sleep(2)
        self._pool = None

    def __delete__(self):
        self._pool.terminate()
        time.sleep(2)
        self._pool = None
