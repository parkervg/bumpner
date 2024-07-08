import sys
from attr import attrs, attrib
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
import re
from textwrap import dedent
import yaml

if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableSequence
else:
    from collections import MutableSequence

from functools import cached_property, reduce
import operator
from bisect import bisect_left
from tabulate import tabulate
from functools import partial
from pathlib import Path
# https://alibaba-cloud.medium.com/why-you-should-use-flashtext-instead-of-regex-for-data-analysis-960a0dc96c6a
from flashtext import KeywordProcessor
import guidance
from guidance.models import Model
from guidance.library import with_temperature
from guidance import select, capture

from .constants import (
    EntityField,
    STOPWORDS,
    OBSERVATION_BEGIN,
    OBSERVATION_END,
    CHILD_ENTITY_SPLIT,
)

WORD_TOKENIZE = re.compile(r"\w+|[^\w\s]+")
ALL_ENTITY_FIELDS = []
for key in EntityField.__annotations__:
    ALL_ENTITY_FIELDS.append(EntityField.__getattribute__(EntityField, key))

tabulate = partial(
    tabulate, headers="keys", showindex="never", tablefmt="simple_outline"
)


def get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)


def update_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    prev_entry = get_by_path(root, items[:-1])[items[-1]]
    if isinstance(prev_entry, dict):
        get_by_path(root, items[:-1])[items[-1]] = value | prev_entry
    else:
        get_by_path(root, items[:-1])[items[-1]] = value


def tokenize(text: str):
    """Simple regex tokenization.
    Pattern adapted from nltk.tokenize.RegexpTokenizer
    """
    return WORD_TOKENIZE.findall(text)


@dataclass
class Entity:
    """Used to hold an entity type and description, from a given taxonomy."""

    name: str
    description: Optional[str] = None

    def __hash__(self):
        return hash(self.name)


@attrs
class Word(str):
    text: str = attrib()
    start_span: int = attrib(init=False)
    label: Union[str, None] = None

    def __repr__(self):
        return self.text

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)


class Text(MutableSequence): # type: ignore
    def __init__(self, *args):
        self.list: List[Word] = list()
        prefix_len = 0
        for arg in list(args):
            arg.start_span = prefix_len
            # + 1 to account for whitespace
            prefix_len += len(arg) + 1
            self.append(arg)

    @classmethod
    def from_lists(cls, words: List[str], labels: Optional[List[str]] = None):
        obj = cls(*map(lambda x: Word(x), words))
        if labels:
            obj.assign_label_by_list(labels)
        return obj

    @classmethod
    def from_string(cls, text: str):
        text = re.sub(r"\s+", " ", text.strip())
        return cls(*[Word(i) for i in tokenize(text)])

    @cached_property
    def string(self):
        return " ".join(self.list)

    @cached_property
    def start_spans(self):
        return [i.start_span for i in self.list]

    def assign_label_by_span(self, span: Tuple[int, int], label: str):
        """Given a span containing (start, end) indices,
        give all `Word` objects in the given index range
        the entity label `label`
        """
        start, end = span
        # Use bisect to find index where these start, end spans
        # should go in this Text object
        start_bisect = bisect_left(self.start_spans, start)
        end_bisect = bisect_left(self.start_spans, end)
        for word in self[start_bisect:end_bisect]:
            word.label = label

    def assign_label_by_list(self, labels: List[str]):
        assert len(labels) == len(self.list)
        for word, label in zip(self.list, labels):
            word.label = label

    def check(self, v):
        # Ensure that we only insert `Word` objects
        if not isinstance(v, Word):
            raise TypeError(v)

    def visualize(self):
        """Utility function to visualize words and their labels.

        Returns:
            ```text
            ┌────────────┬────────────────────────┐
            │ word       │ label                  │
            ├────────────┼────────────────────────┤
            │ today      │ O                      │
            │ ,          │ O                      │
            │ hsdm       │ organization-education │
            │ is         │ O                      │
            │ the        │ O                      │
            │ smallest   │ O                      │
            │ school     │ O                      │
            │ at         │ O                      │
            │ harvard    │ organization-education │
            │ university │ organization-education │
            │ with       │ O                      │
            │ a          │ O                      │
            │ total      │ O                      │
            │ student    │ O                      │
            │ body       │ O                      │
            │ of         │ O                      │
            │ 280        │ O                      │
            │ .          │ O                      │
            └────────────┴────────────────────────┘
            ```
        """
        print(
            tabulate(
                {
                    "word": [i.text for i in self.list],
                    "label": [i.label for i in self.list],
                }
            )
        )

    def render_to_conll(self, sep=" ") -> str:
        """
        Renders text to a 2-column CoNLL-style annotation.
        """
        out = ""
        for word in self.list:
            out += f"{word.text}{sep}{word.label}\n"
        return out.rstrip("\n")

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __str__(self):
        return self.string

    def __repr__(self):
        return self.string


@guidance(stateless=True) # type: ignore
def intro_prompt(lm: Model, entity_objs: List[Entity], few_shot: Optional[str] = None):
    lm += dedent(
        f"""
    Tag each word in the input with an entity type.
    If none apply, use the tag 'O' to denote "no entity". Most words will have this 'O' label.

    Below is a description of each entity type.
    """
    )
    for entity in entity_objs:
        lm += dedent(
            f"""
        {entity.name}: {entity.description}
        """
        )
    if few_shot:
        lm += "\n---\n" + few_shot
    return lm


@guidance(stateless=True) # type: ignore
def inference_prompt(lm: Model, text: Text):
    lm += dedent(
        f"""
        ---
        Input: {text.string}
        Output:
        """
    )
    return lm


@guidance # type: ignore
def constrained_ner(
    lm: Model,
    text: Text,
    entity_label_tree: dict,
    entity_name_to_obj: Dict[str, Entity],
    keyword_to_entity_label: Optional[Dict[str, str]] = None,
    pattern_to_entity_label: Optional[Dict[re.Pattern, str]] = None,
    keyword_processor: Optional[KeywordProcessor] = None,
    skip_stopwords: bool = True,
):
    """This function actually passes the input text to the underlying LLM.
    It's essentially a more fleshed-out version of the NER example
    in the documentation here: https://github.com/guidance-ai/guidance?tab=readme-ov-file#context-free-grammars
    """
    # 1) Assign all labels we can using rules
    #   This limits what gets passed to the language model
    if keyword_to_entity_label is None:
        keyword_to_entity_label = {}
    if pattern_to_entity_label is not None:
        for pattern, label in pattern_to_entity_label.items():
            match = pattern.search(text.string)
            if match:
                text.assign_label_by_span(span=match.span(), label=label)
    if keyword_processor is not None:
        for keyword, start, end in keyword_processor.extract_keywords(
            text.string, span_info=True
        ):
            text.assign_label_by_span(
                span=(start, end), label=keyword_to_entity_label[keyword]
            )
    # 2) Use constrained decoding on the remaining unlabelled words
    #   First predict top-level entity labels, then descend the
    #   entity_label_tree to get child labels.
    top_level_entity_labels = list(entity_label_tree.keys()) + ["O"]
    for word in text:
        str_word: str = word.text
        if word.label is None:  # If we haven't already given the word a label in 1)
            if skip_stopwords and str_word in STOPWORDS:
                # Don't pass stopwords to NER
                lm += str_word + ": " + "O" + "\n"
                continue
            # 2a) Below, we make the first prediction on the top-level entity
            lm += (
                str_word
                + ": "
                + with_temperature(
                    select(top_level_entity_labels, name="curr_entity_label"),
                    temperature=0.0,
                )
            )
            chosen_subtree: Union[dict, None] = entity_label_tree.get(lm["curr_entity_label"], None)
            if chosen_subtree is not None:
                chained_sub_entity_key = f"{lm['curr_entity_label']}"
                while entity_label_tree.get(chained_sub_entity_key, None) is not None:
                    # Predict our sub-entities until we reach a terminal
                    valid_child_entity_names = list(
                        chosen_subtree.keys()
                    )
                    child_entities: List[Entity] = []
                    for c in valid_child_entity_names:
                        child_entities.append(
                            entity_name_to_obj[
                                chained_sub_entity_key + f"{CHILD_ENTITY_SPLIT}{c}"
                            ]
                        )
                    # 2b) Add our child entity descriptions, and make a prediction
                    lm += add_child_entity_observations(child_entities) # type: ignore
                    lm += CHILD_ENTITY_SPLIT + with_temperature(
                        select(valid_child_entity_names, name="curr_entity_label"),
                        temperature=0.0,
                    )
                    chained_sub_entity_key += (
                        f"{CHILD_ENTITY_SPLIT}{lm['curr_entity_label']}"
                    )
            # Finish up this entity prediction with a newline
            lm += "\n"
        else:  # We've already assigned this word a label via a rule
            lm += str_word + ": " + word.label + "\n"
    return lm


@guidance # type: ignore
def add_child_entity_observations(lm: Model, child_entities: List[Entity]):
    """Generates an internal 'thought' containing sub-entity labels and descriptions,
    enclosed in specified begin + end tags (typically xml-style)
    """
    lm += f"\n{OBSERVATION_BEGIN} Now, we can choose from a sub-entity below"
    for entity in child_entities:
        lm += dedent(
            f"""
            {entity.name.split(CHILD_ENTITY_SPLIT)[-1]}: {entity.description}
            """
        )
    lm += f"{OBSERVATION_END}"
    return lm


@attrs
class Bumpner:
    model: Model = attrib()
    yaml_file_or_str: str = attrib()
    few_shot: str = attrib(default="")
    lowercase: bool = attrib(default=True)
    """
    Whether to consider all words in constants.STOPWORDS as 'O' label
    """
    skip_stopwords: bool = attrib(default=True)
    """
    List of all `Entity` dataclasess.
    """
    entity_objs: List[Entity] = attrib(init=False)
    """
    Sort of a tree structure to encode the hierarchical entity labels.
            The values will be null if it's a terminal node.

            Examples:
            ```json
            {
                "PERSON": {
                    "CELEBRITY": {
                        "ACTOR": null,
                        "SINGER": null
                    },
                    "GENERIC_PERSON": null
                },
                "PRODUCT": null,
                "ORG": null,
                "ANUMBER": null
            }
            ```
    """
    entity_label_tree: dict = attrib(init=False)
    keyword_to_entity_label: Dict[str, str] = attrib(init=False)
    pattern_to_entity_label: Dict[re.Pattern, str] = attrib(init=False)
    entity_name_to_obj: Dict[str, Entity] = attrib(init=False)
    keyword_processor: KeywordProcessor = attrib(init=False)

    _partial_model: Model = attrib(init=False)

    def __attrs_post_init__(self):
        self.keyword_to_entity_label = {}
        self.pattern_to_entity_label = {}
        self.entity_objs = []
        self.entity_label_tree = {}
        self.entity_name_to_obj = {}
        try:
            if Path(self.yaml_file_or_str).is_file():
                with open(self.yaml_file_or_str, "r") as f:
                    self.entities: dict = yaml.safe_load(f)
            else:
                self.entities: dict = yaml.safe_load(self.yaml_file_or_str)
        except OSError:
            self.entities: dict = yaml.safe_load(self.yaml_file_or_str)
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.register_entities(entities=self.entities, lowercase=self.lowercase)
        self._partial_model = self.get_partial_model()

    def __call__(self, text: str) -> Text:
        text = Text.from_string(text)
        result = (
            self._partial_model
            + inference_prompt(text=text) # type: ignore
            + capture(
                constrained_ner(
                    text=text,
                    entity_label_tree=self.entity_label_tree,
                    entity_name_to_obj=self.entity_name_to_obj,
                    keyword_processor=self.keyword_processor,
                    keyword_to_entity_label=self.keyword_to_entity_label,
                    pattern_to_entity_label=self.pattern_to_entity_label,
                    skip_stopwords=self.skip_stopwords,
                ), # type: ignore
                "result",
            )
        ).get("result")
        # Remove 'observations' within parentheses
        # TODO: I think there's a way to re-write this in guidance so we only capture relevant chunks
        result = re.sub(
            r"\n{}[^)]*?{}".format(
                re.escape(OBSERVATION_BEGIN), re.escape(OBSERVATION_END)
            ),
            "",
            result,
        )
        words, labels = zip(*[e.split(": ") for e in result.strip().split("\n")])
        return Text.from_lists(words=words, labels=labels)

    def get_partial_model(self) -> Model:
        _model = self.model
        # Only pass top-level entities as intro prompt
        top_level_entities = self.entity_label_tree.keys()
        return _model + intro_prompt(
            [self.entity_name_to_obj.get(name) for name in top_level_entities],
            few_shot=self.few_shot,
        ) # type: ignore

    def register_entities(self, entities: dict, lowercase: bool = True) -> None:
        """Prepare entities into the necessary data structures."""

        def _register_entity(entity_name: str, data: dict):
            entity: Entity = Entity(
                name=entity_name, description=data[EntityField.DESCRIPTION]
            )
            self.entity_objs.append(entity)
            self.entity_name_to_obj[entity_name] = entity
            if (
                CHILD_ENTITY_SPLIT not in entity_name
                and self.entity_label_tree.get(entity_name) is None
            ):
                self.entity_label_tree[entity_name] = None
            # Handle rules
            if EntityField.RULES in data:
                if EntityField.REGEX in data[EntityField.RULES]:
                    for regex_rule in data[EntityField.RULES][EntityField.REGEX]:
                        self.pattern_to_entity_label[
                            re.compile(regex_rule)
                        ] = entity_name
                if EntityField.KEYWORD in data[EntityField.RULES]:
                    for keyword_rule in data[EntityField.RULES][EntityField.KEYWORD]:
                        if lowercase:
                            keyword_rule = keyword_rule.lower()
                        self.keyword_processor.add_keyword(keyword_rule)
                        self.keyword_to_entity_label[keyword_rule] = entity_name

            # Handle case where we have a hierarchical entity
            remaining_fields = set(data.keys()).difference(ALL_ENTITY_FIELDS)
            if remaining_fields:
                for sub_entity_name in remaining_fields:
                    if (
                        CHILD_ENTITY_SPLIT not in entity_name
                        and self.entity_label_tree.get(entity_name) is None
                    ):
                        self.entity_label_tree[entity_name] = {}
                    update_by_path(
                        self.entity_label_tree,
                        entity_name.split(CHILD_ENTITY_SPLIT),
                        {sub_entity_name: None},
                    )
                    _register_entity(
                        entity_name=f"{entity_name}.{sub_entity_name}",
                        data=data[sub_entity_name],
                    )

        for entity_name, data in entities.items():
            _register_entity(entity_name=entity_name, data=data)
