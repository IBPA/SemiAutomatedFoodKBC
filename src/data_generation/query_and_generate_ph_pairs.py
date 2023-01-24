import argparse
from collections import Counter, namedtuple
from datetime import datetime
from itertools import product
from pathlib import Path
import re
import requests
import time
from typing import List, Optional
import sys

import pandas as pd
from pandarallel import pandarallel

FOOD_NAMES_FILEPATH = "../../data/FooDB/foodb_foods.txt"
QUERY_FSTRING = "{} contains"
QUERY_RESULTS_FILEPATH = "../../outputs/data_generation/query_results.txt"
FOOD_PARTS_FILEPATH = "../../data/FoodAtlas/food_parts.txt"
PH_PAIRS_FILEPATH = "../../outputs/data_generation/ph_pairs_{}.txt"
ENTITIES_FILEPATH = "../../data/FoodAtlas/entities.txt"
RELATIONS_FILEPATH = "../../data/FoodAtlas/relations.txt"


CandidateEntity = namedtuple(
    "CandidateEntity",
    [
        "foodatlas_id",
        "type",
        "name",
        "synonyms",
        "other_db_ids",
    ],
    defaults=[
        None,
        None,
        None,
        [],
        {},
    ],
)

CandidateRelation = namedtuple(
    "CandidateRelation",
    [
        "foodatlas_id",
        "name",
        "translation",
    ],
    defaults=[
        None,
        None,
        None,
    ],
)


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query LitSense and generate PH pairs.")

    parser.add_argument(
        "--food_names_filepath",
        type=str,
        default=FOOD_NAMES_FILEPATH,
        help=f"File containing query food names. (Default: {FOOD_NAMES_FILEPATH})",
    )

    parser.add_argument(
        "--query_fstring",
        type=str,
        default=QUERY_FSTRING,
        help=f"fstring to format the query. (Default: {QUERY_FSTRING})",
    )

    parser.add_argument(
        "--query_results_filepath",
        type=str,
        default=QUERY_RESULTS_FILEPATH,
        help=f"Filepath to save the query results. (Default: {QUERY_RESULTS_FILEPATH})",
    )

    parser.add_argument(
        "--food_parts_filepath",
        type=str,
        default=FOOD_PARTS_FILEPATH,
        help=f"Filepath to food parts. (Default: {FOOD_PARTS_FILEPATH})",
    )

    parser.add_argument(
        "--ph_pairs_filepath",
        type=str,
        default=PH_PAIRS_FILEPATH,
        help=f"Filepath to save the PH pairs. (Default: {PH_PAIRS_FILEPATH})",
    )

    parser.add_argument(
        "--entities_filepath",
        type=str,
        default=ENTITIES_FILEPATH,
        help=f"Filepath to save the entities. (Default: {ENTITIES_FILEPATH})",
    )

    parser.add_argument(
        "--relations_filepath",
        type=str,
        default=RELATIONS_FILEPATH,
        help=f"Filepath to save the entities. (Default: {RELATIONS_FILEPATH})",
    )

    args = parser.parse_args()
    return args


def get_food_parts(
    premise: str,
    df_food_parts: pd.DataFrame,
) -> List[CandidateEntity]:
    premise = re.sub('[^A-Za-z0-9 ]+', '', premise.lower()).split()

    parts = []
    for idx, row in df_food_parts.iterrows():
        synonyms = row.food_part_synonyms.split(', ')
        parts_list = [row.food_part] + synonyms

        for part in parts_list:
            if part.lower() in premise:
                part_entity = CandidateEntity(
                    type="food_part",
                    name=row.food_part,
                    synonyms=synonyms,
                )

                if part_entity not in parts:
                    parts.append(part_entity)

    return parts


def merge_candidate_entities(
    candidate_entities: List[CandidateEntity],
    candidates_type: str,
) -> List[CandidateEntity]:

    def _merge_duplicates(duplicates):
        type_ = []
        name = []
        synonyms = []
        other_db_ids = {}
        for d in duplicates:
            if d.foodatlas_id is not None:
                raise ValueError("Candidate entities cannot have foodatlas ID!")
            type_.append(d.type)
            name.append(d.name)
            synonyms.extend(d.synonyms)
            other_db_ids = {**other_db_ids, **d.other_db_ids}

        type_ = list(set(type_))
        name = list(set(name))
        synonyms = list(set(synonyms))

        assert len(type_) == 1

        if type_ == "NCBI_taxonomy":
            assert len(name) == 1
        elif type_ == "MESH":
            if len(name) > 1:
                synonyms = list(set(synonyms + name[1:]))
                name = name[0]

        merged = CandidateEntity(
            type=type_[0],
            name=name[0],
            synonyms=synonyms,
            other_db_ids=other_db_ids,
        )

        for d in duplicates:
            candidate_entities.remove(d)
        candidate_entities.append(merged)

    if candidates_type.split(':')[0] in ["chemical", "organism"]:
        if candidates_type.split(':')[0] == "organism":
            using = "NCBI_taxonomy"
        elif candidates_type.split(':')[0] == "chemical":
            using = "MESH"
        else:
            raise ValueError()

        duplicate_ids = [
            e.other_db_ids[using] for e in candidate_entities if e.other_db_ids[using]]
        duplicate_ids = [x for x, count in Counter(duplicate_ids).items() if count > 1]

        if duplicate_ids:
            for duplicate_id in duplicate_ids:
                duplicates = [
                    e for e in candidate_entities if e.other_db_ids[using] == duplicate_id]
                _merge_duplicates(duplicates)

        return candidate_entities
    elif candidates_type.split(':')[0] == "organism_with_part":
        using = "NCBI_taxonomy"
        unique_ids = {}
        for e in candidate_entities:
            key = e.other_db_ids[using] + e.name.split(' - ')[-1]
            if key in unique_ids:
                unique_ids[key].append(e)
            else:
                unique_ids[key] = [e]

        duplicate_ids = [k for k, v in unique_ids.items() if len(v) > 1]

        for k, v in unique_ids.items():
            if len(v) < 2:
                continue
            _merge_duplicates(v)

        return candidate_entities
    else:
        raise NotImplementedError()


def query_litsense(
    df_food_names: pd.DataFrame,
    query_results_filepath: str,
    df_food_parts: pd.DataFrame,
    query_fstring: str,
    delay: float = 1.0,
) -> Optional[pd.DataFrame]:
    # make outputs dir
    query_results_dir = "/".join(query_results_filepath.split("/")[:-1])
    Path(query_results_dir).mkdir(parents=True, exist_ok=True)

    ncbi_taxonomy_ids = list(set(df_food_names["ncbi_taxonomy_id"].tolist()))

    data = []
    for idx, row in df_food_names.iterrows():
        print(f"Processing {idx+1}/{df_food_names.shape[0]}...")

        food_names = [row["name"], row["name_scientific"]]
        food_names = [x for x in food_names if x != ""]

        for food_name in food_names:
            search_term = query_fstring.format(food_name)
            print(f"Requesting data: {search_term}")

            # avoid throttling
            time.sleep(delay)

            query_url = "https://www.ncbi.nlm.nih.gov/research/litsense-api/api/" + \
                        f"?query={search_term}&rerank=true"

            response = requests.get(query_url)
            if response.status_code != 200:
                raise ValueError(
                    f"Error requesting data from {query_url}: {response.status_code}")

            data_to_extend = []
            print("{} results recieved".format(len(response.json())))
            print("Parsing results...")
            for doc in response.json():
                if doc["annotations"] is None:
                    continue

                r = {}
                r["search_term"] = search_term
                r["pmid"] = doc["pmid"]
                r["pmcid"] = doc["pmcid"]
                r["section"] = doc["section"]
                r["premise"] = doc["text"]
                chemicals = []
                organisms = []

                for ent in doc["annotations"]:
                    ent_split_results = ent.split("|")
                    if len(ent_split_results) != 4:
                        print(f"Unable to parse annotation: {ent}")
                        continue

                    start, n_chars, category, ent_id = ent_split_results
                    start = int(start)
                    n_chars = int(n_chars)

                    if start < 0:
                        print("Start position cannot be less than 0.")
                        continue

                    ent_text = doc["text"][start:(start + n_chars)]

                    if ent_text == "":
                        print(f"Skipping empty entity (start: {start}, n_chars: {n_chars})")
                        continue

                    if ent_id == "None" or ent_id == "-":
                        print(f"Skipping entities with no ID: {ent_text}")
                        continue

                    if category == "species" and not ent_id.isdigit():
                        print(f"Skipping species with non-numerical ID: {ent_id}.")
                        continue

                    if category == "species" and int(ent_id) not in ncbi_taxonomy_ids:
                        print(f"Skipping entity with ID not in FooDB: {ent_text} ({ent_id})")
                        continue

                    if category == "chemical":
                        candidate_ent = CandidateEntity(
                            type="chemical",
                            name=ent_text,
                            other_db_ids={"MESH": ent_id.replace("MESH:", "")}
                        )
                        chemicals.append(candidate_ent)
                    elif category == "species":
                        match = df_food_names[
                            df_food_names["ncbi_taxonomy_id"] == int(ent_id)].iloc[0]
                        synonyms = [ent_text, match["name_scientific"]]
                        synonyms = [
                            x for x in synonyms if x.lower() != match["name"].lower() and x != ""]
                        synonyms = list(set(synonyms))

                        candidate_ent = CandidateEntity(
                            type="organism",
                            name=match["name"],
                            synonyms=synonyms,
                            other_db_ids={"NCBI_taxonomy": ent_id}
                        )
                        organisms.append(candidate_ent)

                if len(chemicals) == 0 or len(organisms) == 0:
                    continue

                # clean up the entities
                chemicals = merge_candidate_entities(chemicals, candidates_type="chemical")
                organisms = merge_candidate_entities(organisms, candidates_type="organism")

                r["chemicals"] = str(chemicals)
                r["organisms"] = str(organisms)
                r["food_parts"] = str(get_food_parts(doc["text"], df_food_parts))
                data_to_extend.append(r)

            data.extend(data_to_extend)

    if not data:
        print("Data empty. Nothing to write.")
        return None

    df = pd.DataFrame(data)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(query_results_filepath, sep='\t', index=False)

    return df


def generate_ph_pairs(
    df: pd.DataFrame,
    ph_pairs_filepath: str,
):
    df["chemicals"] = df["chemicals"].apply(lambda x: eval(x, globals()))
    df["organisms"] = df["organisms"].apply(lambda x: eval(x, globals()))
    df["food_parts"] = df["food_parts"].apply(lambda x: eval(x, globals()))

    contains = CandidateRelation(
        name='contains',
        translation='contains',
    )

    def _f(row):
        newrows = []
        failed = []

        cleaned_premise = " " + re.sub('[^A-Za-z0-9 ]+', ' ', row["premise"].lower()) + ""

        for s, c in product(row["organisms"], row["chemicals"]):
            newrow = row.copy().drop(["search_term", "chemicals", "organisms", "food_parts"])
            newrow["head"] = s
            newrow["relation"] = contains
            newrow["tail"] = c

            organisms = None
            if s.name.lower() not in row["premise"].lower() or \
               f" {s.name.lower()} " not in cleaned_premise:
                for x in s.synonyms:
                    if x.lower() in row["premise"].lower():
                        organisms = x
            else:
                organisms = s.name

            chemicals = None
            if c.name.lower() not in row["premise"].lower():
                for x in c.synonyms:
                    if x.lower() in row["premise"].lower():
                        chemicals = x
            else:
                chemicals = c.name

            if organisms is None or chemicals is None:
                failed.append(row)
                continue

            newrow["hypothesis_string"] = f"{organisms} contains {chemicals}"

            ncbi_taxonomy = s.other_db_ids["NCBI_taxonomy"]
            mesh = c.other_db_ids["MESH"]
            newrow["hypothesis_id"] = f"NCBI_taxonomy:{ncbi_taxonomy}_contains_MESH:{mesh}"
            newrows.append(newrow)

        if row["food_parts"]:
            for s, p, c in product(row["organisms"], row["food_parts"], row["chemicals"]):
                # contains
                newrow = row.copy().drop(["search_term", "chemicals", "organisms", "food_parts"])

                organism_with_part = CandidateEntity(
                    type="organism_with_part",
                    name=f"{s.name} - {p.name}",
                    synonyms=[f"{x} - {p.name}" for x in s.synonyms],
                    other_db_ids=s.other_db_ids,
                )

                newrow["head"] = organism_with_part
                newrow["relation"] = contains
                newrow["tail"] = c

                organisms = None
                if s.name.lower() not in row["premise"].lower() or \
                   f" {s.name.lower()} " not in cleaned_premise:
                    for x in s.synonyms:
                        if x.lower() in row["premise"].lower():
                            organisms = x
                else:
                    organisms = s.name

                parts = None
                if p.name.lower() not in row["premise"].lower():
                    for x in p.synonyms:
                        if x.lower() in row["premise"].lower():
                            parts = x
                else:
                    parts = p.name

                chemicals = None
                if c.name.lower() not in row["premise"].lower():
                    for x in c.synonyms:
                        if x.lower() in row["premise"].lower():
                            chemicals = x
                else:
                    chemicals = c.name

                if organisms is None or parts is None or chemicals is None:
                    failed.append(row)
                    continue

                newrow["hypothesis_string"] = f"{organisms} - {parts} contains {chemicals}"

                ncbi_taxonomy = s.other_db_ids["NCBI_taxonomy"]
                mesh = c.other_db_ids["MESH"]
                newrow["hypothesis_id"] = f"NCBI_taxonomy:{ncbi_taxonomy}-" + \
                                          f"{p.name}_contains_MESH:{mesh}"
                newrows.append(newrow)

        return newrows, failed

    results = []
    skipped = []
    for result, failed in df.parallel_apply(_f, axis=1):
        results.append(pd.DataFrame(result))
        skipped.extend(failed)

    print(f"Skipped {len(skipped)} rows.")
    for x in skipped:
        print(f"Premise: {x['premise']}")
        print(f"Chemicals: {x['chemicals']}")
        print(f"Organisms: {x['organisms']}")
        print(f"Food parts: {x['food_parts']}")

    df_ph_pairs = pd.concat(results).reset_index(drop=True)
    df_ph_pairs.fillna("", inplace=True)
    df_ph_pairs = df_ph_pairs.astype(str)
    df_ph_pairs.drop_duplicates(inplace=True)

    hypothesis_id = df_ph_pairs["hypothesis_id"].tolist()
    duplicates = [item for item, count in Counter(hypothesis_id).items() if count > 1]
    print(f"Found {len(duplicates)} duplicate hypothesis IDs out of {len(hypothesis_id)}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_ph_pairs.to_csv(ph_pairs_filepath.format(timestamp), index=False, sep="\t")

    return df_ph_pairs


def get_food_names_and_parts(
    food_names_filepath: str,
    food_parts_filepath: str,
):
    # food names
    if "data/FooDB" in food_names_filepath:
        data_source = "foodb"
    else:
        raise NotImplementedError()

    if data_source == "foodb":
        df_food_names = pd.read_csv(food_names_filepath, sep='\t', keep_default_na=False)

    # food parts
    df_food_parts = pd.read_csv(food_parts_filepath, sep='\t', keep_default_na=False)

    return df_food_names, df_food_parts


def sample_val_test_set(
    df_ph_pairs: pd.DataFrame,
    val_filepath: str,
    test_filepath: str,
    val_num_premise: float,
    test_num_premise: float,
):
    print(df_ph_pairs)


def main():
    args = parse_argument()

    pandarallel.initialize(progress_bar=True)

    # food names and parts
    df_food_names, df_food_parts = get_food_names_and_parts(
        food_names_filepath=args.food_names_filepath,
        food_parts_filepath=args.food_parts_filepath,
    )

    df = query_litsense(
        df_food_names=df_food_names,
        query_results_filepath=args.query_results_filepath,
        df_food_parts=df_food_parts,
        query_fstring=args.query_fstring,
    )

    df = pd.read_csv(args.query_results_filepath, sep='\t')
    generate_ph_pairs(
        df=df,
        ph_pairs_filepath=args.ph_pairs_filepath,
    )


if __name__ == "__main__":
    main()
