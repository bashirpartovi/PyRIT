# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib
import io
import random
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, TextIO

import pycountry
import requests
from datasets import load_dataset

from pyrit.common.csv_helper import read_csv, write_csv
from pyrit.common.json_helper import read_json, read_jsonl, write_json, write_jsonl
from pyrit.common.path import DATASETS_PATH, DB_DATA_PATH
from pyrit.common.text_helper import read_txt, write_txt
from pyrit.models import (
    QuestionAnsweringDataset,
    QuestionAnsweringEntry,
    QuestionChoice,
    SeedPromptDataset,
)
from pyrit.models.seed_prompt import SeedPrompt

# Define the type for the file handlers
FileHandlerRead = Callable[[TextIO], List[Dict[str, str]]]
FileHandlerWrite = Callable[[TextIO, List[Dict[str, str]]], None]

FILE_TYPE_HANDLERS: Dict[str, Dict[str, Callable]] = {
    "json": {"read": read_json, "write": write_json},
    "jsonl": {"read": read_jsonl, "write": write_jsonl},
    "csv": {"read": read_csv, "write": write_csv},
    "txt": {"read": read_txt, "write": write_txt},
}


def _get_cache_file_name(source: str, file_type: str) -> str:
    """
    Generate a cache file name based on the source URL and file type.
    """
    hash_source = hashlib.md5(source.encode("utf-8")).hexdigest()
    return f"{hash_source}.{file_type}"


def _read_cache(cache_file: Path, file_type: str) -> List[Dict[str, str]]:
    """
    Read data from cache.
    """
    with cache_file.open("r", encoding="utf-8") as file:
        if file_type in FILE_TYPE_HANDLERS:
            return FILE_TYPE_HANDLERS[file_type]["read"](file)
        else:
            valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
            raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")


def _write_cache(cache_file: Path, examples: List[Dict[str, str]], file_type: str):
    """
    Write data to cache.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as file:
        if file_type in FILE_TYPE_HANDLERS:
            FILE_TYPE_HANDLERS[file_type]["write"](file, examples)
        else:
            valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
            raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")


def _fetch_from_public_url(source: str, file_type: str) -> List[Dict[str, str]]:
    """
    Fetch examples from a repository.
    """
    response = requests.get(source)
    if response.status_code == 200:
        if file_type in FILE_TYPE_HANDLERS:
            if file_type == "json":
                return FILE_TYPE_HANDLERS[file_type]["read"](io.StringIO(response.text))
            else:
                return FILE_TYPE_HANDLERS[file_type]["read"](
                    io.StringIO("\n".join(response.text.splitlines()))
                )  # noqa: E501
        else:
            valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
            raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")
    else:
        raise Exception(f"Failed to fetch examples from public URL. Status code: {response.status_code}")


def _fetch_from_file(source: str, file_type: str) -> List[Dict[str, str]]:
    """
    Fetch examples from a local file.
    """
    with open(source, "r", encoding="utf-8") as file:
        if file_type in FILE_TYPE_HANDLERS:
            return FILE_TYPE_HANDLERS[file_type]["read"](file)
        else:
            valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
            raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")


def fetch_examples(
    source: str,
    source_type: Literal["public_url", "file"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """
    Fetch examples from a specified source with caching support.

    Example usage
    >>> examples = fetch_examples(
    >>>     source='https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json',
    >>>     source_type='public_url'
    >>> )

    Args:
        source (str): The source from which to fetch examples.
        source_type (Literal["public_url", "file"]): The type of source ('public_url' or 'file').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        List[Dict[str, str]]: A list of examples.
    """

    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    if not data_home:
        data_home = DB_DATA_PATH / "seed-prompt-entries"
    else:
        data_home = Path(data_home)

    cache_file = data_home / _get_cache_file_name(source, file_type)

    if cache and cache_file.exists():
        return _read_cache(cache_file, file_type)

    if source_type == "public_url":
        examples = _fetch_from_public_url(source, file_type)
    elif source_type == "file":
        examples = _fetch_from_file(source, file_type)

    if cache:
        _write_cache(cache_file, examples, file_type)
    else:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=f".{file_type}") as temp_file:
            FILE_TYPE_HANDLERS[file_type]["write"](temp_file, examples)

    return examples


def fetch_many_shot_jailbreaking_dataset() -> List[Dict[str, str]]:
    """
    Fetch many-shot jailbreaking dataset from a specified source.

    Returns:
        List[Dict[str, str]]: A list of many-shot jailbreaking examples.
    """

    source = "https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/5eac855/examples.json"
    source_type: Literal["public_url"] = "public_url"

    return fetch_examples(source, source_type)


def fetch_seclists_bias_testing_dataset(
    source: str = (
        "https://raw.githubusercontent.com/danielmiessler/SecLists/4e747a4/Ai/LLM_Testing/Bias_Testing/"
        "nationality_geographic_bias.txt"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
    random_seed: Optional[int] = None,
    country: Optional[str] = None,
    region: Optional[str] = None,
    nationality: Optional[str] = None,
    gender: Optional[str] = None,
    skin_color: Optional[str] = None,
) -> SeedPromptDataset:
    """
    Fetch SecLists AI LLM Bias Testing examples from a specified source and create a SeedPromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the SecLists repository Bias_Testing.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.
        random_seed (Optional[int]): Seed for random number generation for reproducibility. Defaults to None.
        country (Optional[str]): Specific country to use for the placeholder. Defaults to None.
        region (Optional[str]): Specific region to use for the placeholder. Defaults to None.
        nationality (Optional[str]): Specific nationality to use for the placeholder. Defaults to None.
        gender (Optional[str]): Specific gender to use for the placeholder. Defaults to None.
        skin_color (Optional[str]): Specific skin color to use for the placeholder. Defaults to None.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples with placeholders replaced.
    """

    if random_seed is not None:
        random.seed(random_seed)

    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    # Define placeholder values generation functions using pycountry and random.choice
    countries = list(pycountry.countries)
    subdivisions = list(pycountry.subdivisions)

    placeholder_values = {
        "Country": lambda: country if country else random.choice(countries).name,
        "Region": lambda: region if region else random.choice(subdivisions).name,
        "Nationality": lambda: (
            nationality if nationality else random.choice(countries).name
        ),  # Same as Country, the target LLM should be able to understand the context
        "GENDER": lambda: gender if gender else random.choice(["male", "female", "non-binary"]),
        "SKIN_COLOR": lambda: skin_color if skin_color else random.choice(["white", "black", "brown", "yellow"]),
    }

    # Replace placeholders in the examples with generated values
    filled_examples = []
    for example in examples:
        prompt = example["prompt"]
        for placeholder, generator in placeholder_values.items():
            values_used = set()
            while f"[{placeholder}]" in prompt:
                value = generator()
                # Ensure the new value is not the same as the previous one
                while value in values_used:
                    value = generator()
                values_used.add(value)
                prompt = prompt.replace(f"[{placeholder}]", value, 1)

        filled_examples.append(prompt)

    # Create SeedPrompt instances from each example in 'filled_examples'
    seed_prompts = [
        SeedPrompt(
            value=example,
            data_type="text",
            name="SecLists Bias Testing Examples",
            dataset_name="SecLists Bias Testing Examples",
            harm_categories=["bias_testing"],
            description="A dataset of SecLists AI LLM Bias Testing examples with placeholders replaced.",
        )
        for example in filled_examples
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)

    return seed_prompt_dataset


def fetch_xstest_dataset(
    source: str = "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/a3bb396/xstest_v2_prompts.csv",
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> SeedPromptDataset:
    """
    Fetch XSTest examples and create a SeedPromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the exaggerated-safety repository.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/paul-rottger/exaggerated-safety
    """

    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    # Extract prompts, harm categories, and other relevant data from the fetched examples
    prompts = [example["prompt"] for example in examples]
    harm_categories = [example["note"] for example in examples]

    seed_prompts = [
        SeedPrompt(
            value=example,
            data_type="text",
            name="XSTest Examples",
            dataset_name="XSTest Examples",
            harm_categories=harm_categories,
            description="A dataset of XSTest examples containing various categories such as violence, drugs, etc.",
        )
        for example in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)

    return seed_prompt_dataset


def fetch_harmbench_dataset(
    source: str = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/behavior_datasets/"
        "harmbench_behaviors_text_all.csv"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
) -> SeedPromptDataset:
    """
    Fetch HarmBench examples and create a SeedPromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the HarmBench repository.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/centerforaisafety/HarmBench
    """

    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    # Required keys to validate each example
    required_keys = {"Behavior", "SemanticCategory"}

    # Initialize containers for prompts and semantic categories
    prompts = []
    semantic_categories = set()

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    # Validate each example and extract data
    for example in examples:
        # Check for missing keys in the example
        missing_keys = required_keys - example.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in example: {', '.join(missing_keys)}")

        # Extract and append the data to respective containers
        prompts.append(example["Behavior"])
        semantic_categories.add(example["SemanticCategory"])

    seed_prompts = [
        SeedPrompt(
            value=example,
            data_type="text",
            name="HarmBench Examples",
            dataset_name="HarmBench Examples",
            harm_categories=list(semantic_categories),
            description="A dataset of HarmBench examples containing various categories such as chemical,"
            "biological, illegal activities, etc.",
        )
        for example in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)

    return seed_prompt_dataset


def fetch_decoding_trust_stereotypes_dataset(
    source: str = (
        "https://raw.githubusercontent.com/AI-secure/DecodingTrust/main/data/stereotype/dataset/user_prompts.csv"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
    stereotype_topics: Optional[List[str]] = None,
    target_groups: Optional[List[str]] = None,
    system_prompt_type: Literal["benign", "untargeted", "targeted"] = "targeted",
) -> SeedPromptDataset:
    """
    Fetch DecodingTrust examples and create a SeedPromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the DecodingTrust repository.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.
        stereotype_topics (Optional[List[str]]): List of stereotype topics to filter the examples. Defaults to None.
            The list of all 16 stereotype_topics can be found here:
            https://github.com/AI-secure/DecodingTrust/tree/main/data/stereotype
            Defaults to None, which means all topics are included.
        target_groups (Optional[List[str]]): List of target groups to filter the examples. Defaults to None.
            The list of all 24 target_groups can be found here:
            https://github.com/AI-secure/DecodingTrust/tree/main/data/stereotype
            Defaults to None, which means all target groups are included.
        system_prompt_type (Literal["benign", "untargeted", "targeted"]): The type of system prompt to use.
            Defaults to "targeted".

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/centerforaisafety/HarmBench
    """
    if system_prompt_type not in ["benign", "untargeted", "targeted"]:
        raise ValueError("Invalid system_prompt_type. Expected 'benign', 'untargeted', or 'targeted'}")
    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    if system_prompt_type == "targeted":
        system_prompt = SeedPrompt.from_yaml_file(
            Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "dt_stereotypes_targeted.yaml"
        )
    elif system_prompt_type == "untargeted":
        system_prompt = SeedPrompt.from_yaml_file(
            Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "dt_stereotypes_untargeted.yaml"
        )
    else:
        system_prompt = SeedPrompt.from_yaml_file(
            Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "dt_stereotypes_benign.yaml"
        )

    # Required keys to validate each example
    required_keys = {"stereotype_topic", "target_group", "user_prompt"}

    # Initialize containers for prompts and semantic categories
    prompts = []

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    is_targeted = system_prompt_type == "targeted"
    # Validate each example and extract data
    for example in examples:
        # Check for missing keys in the example
        missing_keys = required_keys - example.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in example: {', '.join(missing_keys)}")
        # Check if the example should be filtered based on the provided topics and target groups
        if (
            stereotype_topics
            and example["stereotype_topic"] not in stereotype_topics
            or (target_groups and example["target_group"] not in target_groups)
        ):
            continue

        prompt = (
            system_prompt.render_template_value(prompt=example["user_prompt"], target_group=example["target_group"])
            if is_targeted
            else system_prompt.render_template_value(prompt=example["user_prompt"])
        )
        prompts.append(prompt)

    seed_prompts = [
        SeedPrompt(
            value=prompt,
            data_type="text",
            name="DecodingTrust Stereotypes Examples",
            dataset_name="DecodingTrust Stereotypes Examples",
            harm_categories=["stereotypes"],
            description="""A dataset of Decoding Trust 'Stereotypes' example prompts across many different topics
            and target groups. These prompts attempt to convince the target to agree with a harmful
            stereotype.""",
        )
        for prompt in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset


def fetch_adv_bench_dataset(
    cache: bool = True,
    main_categories: Optional[
        List[
            Literal[
                "Autonomy",
                "Physical",
                "Psychological",
                "Reputational",
                "Financial and Business",
                "Human Rights and Civil Liberties",
                "Societal and Cultural",
                "Political and Economic",
                "Environmental",
            ]
        ]
    ] = None,
    sub_categories: Optional[List[str]] = None,
) -> SeedPromptDataset:
    """
    Retrieve AdvBench examples enhanced with categories from a collaborative and human-centered harms taxonomy.

    This function fetches a dataset extending the original AdvBench Dataset by adding harm types to each prompt.
    Categorization was done using the Claude 3.7 model based on the Collaborative, Human-Centered Taxonomy of AI,
    Algorithmic, and Automation Harms (https://arxiv.org/abs/2407.01294v2). Each entry includes at least one main
    category and one subcategory to enable better filtering and analysis of the dataset.

    Useful link: https://arxiv.org/html/2407.01294v2/x2.png (Overview of the Harms Taxonomy)

    Args:
        cache (bool): Whether to cache the fetched examples. Defaults to True.

        main_categories (Optional[List[str]]): A list of main harm categories to search for in the dataset.
            For descriptions of each category, see the paper: arXiv:2407.01294v2
            Defaults to None, which includes all 9 main categories.

        sub_categories (Optional[List[str]]): A list of harm subcategories to search for in the dataset.
            For the complete list of all subcategories, see the paper: arXiv:2407.01294v2.
            Defaults to None, which includes all subcategories.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench. Based on research in paper:
        https://arxiv.org/abs/2307.15043 written by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr,
        J. Zico Kolter, Matt Fredrikson.

        The categorization approach was proposed by @paulinek13, who suggested using the Collaborative, Human-Centred
        Taxonomy of AI, Algorithmic, and Automation Harms (arXiv:2407.01294v2) to classify the AdvBench examples and
        used Anthropic's Claude 3.7 Sonnet model to perform the categorization based on the taxonomy's descriptions.
    """
    dataset = fetch_examples(
        source=str(Path(DATASETS_PATH) / "data" / "adv_bench_dataset.json"), source_type="file", cache=cache
    )

    filtered = dataset["data"]  # type: ignore

    if main_categories or sub_categories:
        main_set = set(main_categories or [])
        sub_set = set(sub_categories or [])

        # Include an entry if it matches ANY specified main category OR ANY specified subcategory
        filtered = [
            item
            for item in filtered
            if (main_set and any(cat in main_set for cat in item["main_categories"]))
            or (sub_set and any(cat in sub_set for cat in item["sub_categories"]))
        ]

    seed_prompts = [
        SeedPrompt(
            value=item["prompt"],
            data_type="text",
            name="AdvBench Dataset [Extended]",
            dataset_name="AdvBench Dataset",
            harm_categories=item["main_categories"] + item["sub_categories"],
            description="""AdvBench is a set of 520 harmful behaviors formulated as instructions. This dataset
            has been extended to include harm categories for better filtering and analysis. The adversary's goal
            is instead to find a single attack string that will cause the model to generate any response that
            attempts to comply with the instruction, and to do so over as many harmful behaviors as possible.""",
            source="https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench",
        )
        for item in filtered
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset


def fetch_pku_safe_rlhf_dataset(
    include_safe_prompts: bool = True,
    filter_harm_categories: Optional[
        List[
            Literal[
                "Animal Abuse",
                "Copyright Issues",
                "Cybercrime",
                "Discriminatory Behavior",
                "Disrupting Public Order",
                "Drugs",
                "Economic Crime",
                "Endangering National Security",
                "Endangering Public Health",
                "Environmental Damage",
                "Human Trafficking",
                "Insulting Behavior",
                "Mental Manipulation",
                "Physical Harm",
                "Privacy Violation",
                "Psychological Harm",
                "Sexual Content",
                "Violence",
                "White-Collar Crime",
            ]
        ]
    ] = None,
) -> SeedPromptDataset:
    """
    Fetch PKU-SafeRLHF examples and create a SeedPromptDataset.

    Args:
        include_safe_prompts (bool): All prompts in the dataset are returned if True; the dataset has
            RLHF markers for unsafe responses, so if False we only return the unsafe subset.
        filter_harm_categories: List of harm categories to filter the examples.
            Defaults to None, which means all categories are included.
            Otherwise, only prompts with at least one matching category are included.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF. Based on research in paper:
        https://arxiv.org/pdf/2406.15513 written by Jiaming Ji and Donghai Hong and Borong Zhang and Boyuan Chen
        and Josef Dai and Boren Zheng and Tianyi Qiu and Boxun Li and Yaodong Yang
    """
    data = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")

    seed_prompts = []

    for item in data["train"]:
        is_unsafe = not (item["is_response_0_safe"] and item["is_response_1_safe"])

        # Skip if safe prompts are excluded and the item is safe
        if not include_safe_prompts and not is_unsafe:
            continue

        # Collect harm categories from both responses, avoid duplicates
        harm_categories = {
            category
            for response_key in ["response_0_harm_category", "response_1_harm_category"]
            for category, is_harmful in item[response_key].items()
            if is_harmful
        }

        # Add the prompt in two cases:
        # 1. No harm categories are provided for filtering
        # 2. At least one harm category matches the specified filter criteria
        if not filter_harm_categories or any(category in filter_harm_categories for category in harm_categories):
            seed_prompts.append(
                SeedPrompt(
                    value=item["prompt"],
                    data_type="text",
                    name="PKU-SafeRLHF",
                    dataset_name="PKU-SafeRLHF",
                    harm_categories=sorted(list(harm_categories)),
                    description="""This is a Hugging Face dataset that labels a prompt and 2 responses categorizing
                        their helpfulness or harmfulness. Only the 'prompt' column is extracted.""",
                    source="https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF",
                )
            )

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset


def fetch_llm_latent_adversarial_training_harmful_dataset() -> SeedPromptDataset:
    data = load_dataset("LLM-LAT/harmful-dataset", "default")

    prompts = [item["prompt"] for item in data["train"]]

    # Create SeedPrompt instances from each example in 'prompts'
    seed_prompts = [
        SeedPrompt(
            value=prompt,
            data_type="text",
            name="LLM-LAT/harmful-dataset",
            dataset_name="LLM-LAT/harmful-dataset",
            description="This dataset contains prompts used to assess and analyze harmful behaviors in llm",
            source="https://huggingface.co/datasets/LLM-LAT/harmful-dataset",
        )
        for prompt in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset


def fetch_tdc23_redteaming_dataset() -> SeedPromptDataset:
    """
    Fetch TDC23-RedTeaming examples and create a SeedPromptDataset.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.
    """
    # Load the TDC23-RedTeaming dataset
    data = load_dataset("walledai/TDC23-RedTeaming", "default")

    prompts = [item["prompt"] for item in data["train"]]

    # Create SeedPrompt instances from each example in 'prompts'
    seed_prompts = [
        SeedPrompt(
            value=prompt,
            data_type="text",
            name="walledai/TDC23-RedTeaming",
            dataset_name="walledai/TDC23-RedTeaming",
            description="""TDC23-RedTeaming dataset from HuggingFace,
                    created by Walled AI (https://huggingface.co/walledai).
                    Contains 100 prompts aimed at generating harmful content
                    across multiple harm categories related to fairness,
                    misinformation, dangerous and criminal activities,
                    violence, etc. in the style of writing narratives.""",
            source="https://huggingface.co/datasets/walledai/TDC23-RedTeaming",
        )
        for prompt in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset


def fetch_forbidden_questions_dataset() -> SeedPromptDataset:
    """
    Fetch Forbidden question dataset and return it as a SeedPromptDataset

    Returns: SeedPromptDataset

    Note: For more info
        Paper - https://arxiv.org/abs/2308.03825
        Github - https://github.com/verazuo/jailbreak_llms/
        Website - https://jailbreak-llms.xinyueshen.me/
    """
    data = load_dataset("TrustAIRLab/forbidden_question_set", "default")

    authors = ["Xinyue Shen", "Zeyuan Chen", "Michael Backes", "Yun Shen", "Yang Zhang"]
    seed_prompts = [
        SeedPrompt(
            value=item["question"],
            data_type="text",
            name="TrustAIRLab/forbidden_question_set",
            dataset_name="TrustAIRLab/forbidden_question_set",
            authors=authors,
            harm_categories=item["content_policy_name"],
            source="https://huggingface.co/datasets/TrustAIRLab/forbidden_question_set",
            description="""This is the Forbidden Question Set dataset proposed in the ACM CCS 2024 paper
        "Do Anything Now'': Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models.
        It contains 390 questions (= 13 scenarios x 30 questions) adopted from OpenAI Usage Policy.
        The focus is on 13 scenarios, including Illegal Activity, Hate Speech, Malware Generation,
        Physical Harm, Economic Harm, Fraud, Pornography, Political Lobbying, Privacy Violence, Legal Opinion,
        Financial Advice, Health Consultation, and Government Decision.""",
        )
        for item in data["train"]
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset


def fetch_librAI_do_not_answer_dataset() -> SeedPromptDataset:
    """
    Fetch the LibrAI 'Do Not Answer' dataset and return it as a SeedPromptDataset.

    Relevant Columns:
    - "risk_area"
    - "types_of_harm"
    - "specific_harms"
    - "question"

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.

    Note: For more info
    - Paper       - https://arxiv.org/abs/2308.13387
    - Github      - https://github.com/libr-ai/do-not-answer
    - HF Dataset  - https://huggingface.co/datasets/LibrAI/do-not-answer
    """
    # Load dataset from Hugging Face
    data = load_dataset("LibrAI/do-not-answer", split="train")

    seed_prompts = [
        SeedPrompt(
            value=entry["question"],
            data_type="text",
            name="",
            dataset_name="LibrAI/Do-Not-Answer",
            harm_categories=[entry["risk_area"], entry["types_of_harm"], entry["specific_harms"]],
            description=(
                f"This is a prompt from the 'Do Not Answer' dataset under the risk area: {entry['risk_area']}, "
                f"harm type: {entry['types_of_harm']}, and specific harm: {entry['specific_harms']}."
            ),
            source="https://huggingface.co/datasets/LibrAI/do-not-answer",
        )
        for entry in data
    ]

    # Create a SeedPromptDataset from the list of SeedPrompt instances
    return SeedPromptDataset(prompts=seed_prompts)


def fetch_wmdp_dataset(category: Optional[str] = None) -> QuestionAnsweringDataset:
    """
    Fetch WMDP examples and create a QuestionAnsweringDataset.

    Args:
        category (str): The dataset category, one of "cyber", "bio", "chem"

    Returns:
        QuestionAnsweringDataset: A QuestionAnsweringDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://huggingface.co/datasets/cais/wmdp
    """

    # Determine which subset of data to load
    data_categories = None
    if not category:  # if category is not specified, read in all 3 subsets of data
        data_categories = ["wmdp-cyber", "wmdp-bio", "wmdp-chem"]
    elif category not in ["cyber", "bio", "chem"]:
        raise ValueError(f"Invalid Parameter: {category}. Expected 'cyber', 'bio', or 'chem'")
    else:
        data_categories = ["wmdp-" + category]

    # Read in cybersecurity dataset
    questions_answers = []
    for name in data_categories:
        ds = load_dataset("cais/wmdp", name)
        for i in range(0, len(ds["test"])):
            # For each question, save the 4 possible choices and their respective index
            choices = []
            for j in range(0, 4):
                c = QuestionChoice(index=j, text=ds["test"]["choices"][i][j])
                choices.append(c)

            entry = QuestionAnsweringEntry(
                question=ds["test"]["question"][i],
                answer_type="int",
                correct_answer=ds["test"]["answer"][i],
                choices=choices,
            )
            questions_answers.append(entry)

    dataset = QuestionAnsweringDataset(
        name="wmdp",
        description="""The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning. The Weapons of Mass
        Destruction Proxy (WMDP) benchmark is a dataset of 4,157 multiple-choice questions surrounding hazardous
        knowledge in biosecurity, cybersecurity, and chemical security. WMDP serves as both a proxy evaluation
        for hazardous knowledge in large language models (LLMs) and a benchmark for unlearning methods to remove such
        hazardous knowledge. To guide progress on mitigating risk from LLMs, we develop CUT, a state-of-the-art
        unlearning method which reduces model performance on WMDP while maintaining general language model
        capabilities.""",
        author="""Nathaniel Li and Alexander Pan and Anjali Gopal and Summer Yue and Daniel Berrios and Alice
        Gatti and Justin D. Li and Ann-Kathrin Dombrowski and Shashwat Goel and Long Phan and Gabriel Mukobi
        and Nathan Helm-Burger and Rassin Lababidi and Lennart Justen and Andrew B. Liu and Michael Chen and
        Isabelle Barrass and Oliver Zhang and Xiaoyuan Zhu and Rishub Tamirisa and Bhrugu Bharathi and Adam Khoja
        and Zhenqi Zhao and Ariel Herbert-Voss and Cort B. Breuer and Andy Zou and Mantas Mazeika and Zifan Wang
        and Palash Oswal and Weiran Liu and Adam A. Hunt and Justin Tienken-Harder and Kevin Y. Shih and Kemper
        Talley and John Guan and Russell Kaplan and Ian Steneker and David Campbell and Brad Jokubaitis and
        Alex Levinson and Jean Wang and William Qian and Kallol Krishna Karmakar and Steven Basart and Stephen
        Fitz and Mindy Levine and Ponnurangam Kumaraguru and Uday Tupakula and Vijay Varadharajan and Yan
        Shoshitaishvili and Jimmy Ba and Kevin M. Esvelt and Alexandr Wang and Dan Hendrycks""",
        source="https://huggingface.co/datasets/cais/wmdp",
        questions=questions_answers,
    )

    return dataset


def fetch_aya_redteaming_dataset(
    cache: bool = True,
    data_home: Optional[Path] = None,
    language: Literal["English", "Hindi", "French", "Spanish", "Arabic", "Russian", "Serbian", "Tagalog"] = "English",
    harm_categories: Optional[
        List[
            Literal[
                "Bullying & Harassment",
                "Discrimination & Injustice",
                "Graphic material",
                "Harms of Representation Allocation and Quality of Service",
                "Hate Speech",
                "Non-consensual sexual content",
                "Profanity",
                "Self-Harm",
                "Violence, Threats & Incitement",
            ]
        ]
    ] = None,
    harm_scope: Optional[Literal["global", "local"]] = None,
) -> SeedPromptDataset:
    """
    Fetch examples from the Aya Red-teaming dataset with optional filtering and create a SeedPromptDataset.

    Args:
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home: Directory to store cached data. Defaults to None.
        language: The language to fetch prompts for. Defaults to "English".
        harm_categories: List of harm categories to filter the examples.
            Defaults to None, which means all categories are included.
            Otherwise, only prompts with at least one matching category are included.
        harm_scope: Whether to fetch globally or locally harmful prompts.
            Defaults to None, which means all examples are included.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the filtered examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://huggingface.co/datasets/CohereForAI/aya_redteaming/blob/main/README.md \n
        Related paper: https://arxiv.org/abs/2406.18682 \n
        The dataset license: Apache 2.0

    Warning:
        Due to the nature of these prompts, it may be advisable to consult your relevant legal
        department before testing them with LLMs to ensure compliance and reduce potential risks.
    """
    _lang = {
        "English": "eng",
        "Hindi": "hin",
        "French": "fra",
        "Spanish": "spa",
        "Arabic": "arb",
        "Russian": "rus",
        "Serbian": "srp",
        "Tagalog": "tgl",
    }

    examples = fetch_examples(
        source=f"https://huggingface.co/datasets/CohereForAI/aya_redteaming/raw/main/aya_{_lang[language]}.jsonl",
        source_type="public_url",
        cache=cache,
        data_home=data_home,
    )

    seed_prompts = []

    for example in examples:
        categories = eval(example["harm_category"])
        if harm_categories is None or any(cat in categories for cat in harm_categories):
            if harm_scope is None or example["global_or_local"] == harm_scope:
                seed_prompts.append(
                    SeedPrompt(
                        value=example["prompt"],
                        data_type="text",
                        name="Aya Red-teaming Examples",
                        dataset_name="Aya Red-teaming Examples",
                        harm_categories=categories,
                        source="https://huggingface.co/datasets/CohereForAI/aya_redteaming",
                    )
                )

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset


def fetch_babelscape_alert_dataset(
    category: Literal["alert", "alert_adversarial"] = "alert_adversarial"
) -> SeedPromptDataset:
    """
    Fetch the Babelscape/ALERT dataset and create a SeedPromptDataset.

    Args:
        category (str): The dataset category, "alert" or "alert_adversarial"

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples.
    """

    data_categories = None
    if not category:  # if category is not specified, read both subsets
        data_categories = ["alert_adversarial", "alert"]
    elif category not in ["alert_adversarial", "alert"]:
        raise ValueError(f"Invalid Parameter: {category}. Expected 'alert_adversarial' or 'alert'")
    else:
        data_categories = [category]

    # Load specified subset or both catagories
    for name in data_categories:
        data = load_dataset("Babelscape/ALERT", name)
        prompts = [item["prompt"] for item in data["test"]]

    # Create SeedPrompt instances from each example in 'prompts'
    seed_prompts = [
        SeedPrompt(
            value=prompt,
            data_type="text",
            name="",
            dataset_name="Babelscape/ALERT",
            description="""ALERT by Babelscape is a dataset that consists
            of two different categories, 'alert' with 15k red teaming prompts,
            and 'alert_adversarial' with 30k adversarial red teaming prompts.""",
            source="https://huggingface.co/datasets/Babelscape/ALERT",
        )
        for prompt in prompts
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
