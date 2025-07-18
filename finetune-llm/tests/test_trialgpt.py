import json
import json
from nltk.tokenize import sent_tokenize
import os
import openai
import time


def parse_criteria(criteria):
    output = ""
    criteria = criteria.split("\n\n")

    idx = 0
    for criterion in criteria:
        criterion = criterion.strip()

        if (
            "inclusion criteria" in criterion.lower()
            or "exclusion criteria" in criterion.lower()
        ):
            continue

        if len(criterion) < 5:
            continue

        output += f"{idx}. {criterion}\n"
        idx += 1

    return output


def print_trial(
    trial_info: dict,
    inc_exc: str,
) -> str:
    """Given a dict of trial information, returns a string of trial."""

    trial = f"Title: {trial_info['brief_title']}\n"
    trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
    trial += f"Interventions: {', '.join(trial_info['drugs_list'])}\n"
    trial += f"Summary: {trial_info['brief_summary']}\n"

    if inc_exc == "inclusion":
        trial += "Inclusion criteria:\n %s\n" % parse_criteria(
            trial_info["inclusion_criteria"]
        )
    elif inc_exc == "exclusion":
        trial += "Exclusion criteria:\n %s\n" % parse_criteria(
            trial_info["exclusion_criteria"]
        )

    return trial


def get_matching_prompt(
    trial_info: dict,
    inc_exc: str,
    patient: str,
):
    """Output the prompt."""
    prompt = f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the {inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level.\n"

    if inc_exc == "inclusion":
        prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

    elif inc_exc == "exclusion":
        prompt += "The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

    prompt += f"You should check the {inc_exc} criteria one-by-one, and output the following three elements for each criterion:\n"
    prompt += f"\tElement 1. For each {inc_exc} criterion, briefly generate your reasoning process: First, judge whether the criterion is not applicable (not very common), where the patient does not meet the premise of the criterion. Then, check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet the criterion. If there is no direct evidence, try to infer from existing evidence, and answer one question: If the criterion is true, is it possible that a good patient note will miss such information? If impossible, then you can assume that the criterion is not true. Otherwise, there is not enough information.\n"
    prompt += f"\tElement 2. If there is relevant information, you must generate a list of relevant sentence IDs in the patient note. If there is no relevant information, you must annotate an empty list.\n"
    prompt += f"\tElement 3. Classify the patient eligibility for this specific {inc_exc} criterion: "

    if inc_exc == "inclusion":
        prompt += 'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "included" denotes that the patient meets the inclusion criterion, while "not included" means the reverse.\n'
    elif inc_exc == "exclusion":
        prompt += 'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "excluded" denotes that the patient meets the exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.\n'

    prompt += "You should output only a JSON dict exactly formatted as: dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}."

    user_prompt = f"Here is the patient note, each sentence is led by a sentence_id:\n{patient}\n\n"
    user_prompt += (
        f"Here is the clinical trial:\n{print_trial(trial_info, inc_exc)}\n\n"
    )
    user_prompt += f"Plain JSON output:"

    return prompt, user_prompt


def trialgpt_matching(trial: dict, patient: str, model: str, client):
    results = {}

    # doing inclusions and exclusions in separate prompts
    for inc_exc in ["inclusion", "exclusion"]:
        system_prompt, user_prompt = get_matching_prompt(trial, inc_exc, patient)

        results[inc_exc + "_system_prompt"] = system_prompt
        results[inc_exc + "_user_prompt"] = user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        end = time.time()
        execution_delay = end - start
        results[inc_exc + "_inference_time"] = execution_delay

        message = response.choices[0].message.content.strip()
        message = message.strip("`").strip("json")

        try:
            results[inc_exc] = json.loads(message)
        except:
            results[inc_exc] = message

    return results


def test_trialgpt():
    client = openai.OpenAI(
        base_url="http://localhost:8998/v1", api_key="sk-no-key-required"
    )

    corpus = "trec_2021"
    model = "Phi-3-medium-4k-instruct"
    # model = "SOLAR-10.7B-Instruct"

    dataset = json.load(open(f"dataset/{corpus}/retrieved_trials.json"))

    output_path = f"results/matching_results_{corpus}_{model}.json"

    # Dict{Str(patient_id): Dict{Str(label): Dict{Str(trial_id): Str(output)}}}
    if os.path.exists(output_path):
        output = json.load(open(output_path))
    else:
        output = {}

    trec2021_patient_ids = [10, 30, 34, 60]
    for patient_id in trec2021_patient_ids:
        instance = dataset[patient_id]

        # Dict{'patient': Str(patient), '0': Str(NCTID), ...}
        patient_id = instance["patient_id"]
        patient = instance["patient"]
        sents = sent_tokenize(patient)
        sents.append(
            "The patient will provide informed consent, and will comply with the trial protocol without any practical issues."
        )
        sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
        patient = "\n".join(sents)

        # initialize the patient id in the output

        if patient_id not in output:
            output[patient_id] = {"0": {}, "1": {}, "2": {}}

        for label in ["2", "1", "0"]:
            if label not in instance:
                continue

            for trial in instance[label]:
                trial_id = trial["NCTID"]

                # already calculated and cached
                if trial_id in output[patient_id][label]:
                    continue

                # in case anything goes wrong (e.g., API calling errors)
                try:
                    results = trialgpt_matching(trial, patient, model, client)
                    output[patient_id][label][trial_id] = results

                    with open(output_path, "w") as f:
                        json.dump(output, f, indent=4)

                except Exception as e:
                    print(e)
                    continue

                break
