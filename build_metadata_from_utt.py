import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse utt_id into speaker metadata and save to CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file containing utt_id. Supports CSV with utt_id column or SCP text file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metadata_from_utt.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--speakers_txt",
        type=str,
        default=None,
        help="Optional LibriSpeech SPEAKERS.TXT path for gender lookup.",
    )
    return parser.parse_args()


def strip_wav_suffix(utt_id):
    return utt_id[:-4] if utt_id.endswith(".wav") else utt_id


def parse_utt_id(utt_id):
    """
    Example:
    4446-2273-0032_4970-29093-0020.wav
    -> spk1_id=4446, spk2_id=4970
    """
    utt_core = strip_wav_suffix(utt_id)
    parts = utt_core.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected utt_id format: {utt_id}")

    src1, src2 = parts
    src1_fields = src1.split("-")
    src2_fields = src2.split("-")
    if len(src1_fields) < 3 or len(src2_fields) < 3:
        raise ValueError(f"Unexpected source format inside utt_id: {utt_id}")

    return {
        "utt_id": utt_id,
        "src1_utt": src1,
        "src2_utt": src2,
        "spk1_id": src1_fields[0],
        "spk2_id": src2_fields[0],
        "chap1_id": src1_fields[1],
        "chap2_id": src2_fields[1],
    }


def normalize_gender_pair(gender1, gender2):
    if not gender1 or not gender2:
        return ""
    pair = "".join(sorted([gender1, gender2]))
    return pair


def load_speakers_txt(speakers_txt_path):
    speaker_to_gender = {}
    with open(speakers_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if "|" not in line:
                continue
            fields = [field.strip() for field in line.split("|")]
            if len(fields) < 2:
                continue
            speaker_id = fields[0]
            gender = fields[1]
            if speaker_id.isdigit() and gender in {"M", "F"}:
                speaker_to_gender[speaker_id] = gender
    return speaker_to_gender


def read_utt_ids_from_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "utt_id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV does not contain utt_id column: {csv_path}")
        for row in reader:
            utt_id = row["utt_id"].strip()
            if utt_id:
                rows.append(row)
    return rows


def read_utt_ids_from_scp(scp_path):
    rows = []
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id = line.split()[0]
            rows.append({"utt_id": utt_id})
    return rows


def read_utt_ids(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        return read_utt_ids_from_csv(input_path)
    return read_utt_ids_from_scp(input_path)


def main():
    args = parse_args()
    input_rows = read_utt_ids(args.input)

    speaker_to_gender = {}
    if args.speakers_txt:
        speaker_to_gender = load_speakers_txt(args.speakers_txt)

    rows = []
    for input_row in input_rows:
        utt_id = input_row["utt_id"].strip()
        parsed = parse_utt_id(utt_id)
        row = {
            "utt_id": parsed["utt_id"],
            "sisdr": input_row.get("sisdr", ""),
            "energy_gap_db": input_row.get("energy_gap_db", ""),
            "spk1_id": parsed["spk1_id"],
            "spk2_id": parsed["spk2_id"],
            "gender_pair": "",
        }
        if speaker_to_gender:
            gender1 = speaker_to_gender.get(parsed["spk1_id"], "")
            gender2 = speaker_to_gender.get(parsed["spk2_id"], "")
            row["gender_pair"] = normalize_gender_pair(gender1, gender2)
        rows.append(row)

    fieldnames = ["utt_id", "sisdr", "energy_gap_db", "spk1_id", "spk2_id", "gender_pair"]

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} utterances.")
    print(f"Saved metadata CSV to: {args.output}")


if __name__ == "__main__":
    main()
