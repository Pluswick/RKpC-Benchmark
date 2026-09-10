"""Build aggregate source-provenance metadata for the private Kp_Data table.

The source table and record-level values are intentionally not written.  The
output contains bibliographic facts and aggregate contribution counts only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
from rdkit import RDLogger


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prepare_data  # noqa: E402


TISSUES = prepare_data.TISSUES


SOURCES = [
    {
        "source_id": "KPD01",
        "citation": "Jansson R, Bredberg U, Ashton M (2008) Prediction of drug tissue to plasma concentration ratios using a measured volume of distribution in combination with lipophilicity. J Pharm Sci 97:2324-2339.",
        "doi_or_url": "https://doi.org/10.1002/jps.21130",
        "source_type": "journal article",
        "workbook_key": "Prediction of Drug Tissue to Plasma Concentration Ratios Using a Measured Volume of Distribution in Combination With Lipophilicity",
    },
    {
        "source_id": "KPD02",
        "citation": "Bj\u00f6rkman S (2002) Prediction of the volume of distribution of a drug: which tissue-plasma partition coefficients are needed? J Pharm Pharmacol 54:1237-1245.",
        "doi_or_url": "https://doi.org/10.1211/002235702320402080",
        "source_type": "journal article",
        "workbook_key": "Prediction of the volume of distribution of a drug: which tissue-plasma partition coefficients are needed?",
    },
    {
        "source_id": "KPD03",
        "citation": "Handa K, Yoshimura S, Kageyama M, Iijima T (2024) Development of novel methods for QSAR modeling by machine learning repeatedly: a case study on drug distribution to each tissue. J Chem Inf Model 64:3662-3669.",
        "doi_or_url": "https://doi.org/10.1021/acs.jcim.4c00046",
        "source_type": "journal article",
        "workbook_key": "Development of Novel Methods for QSAR Modeling by Machine Learning Repeatedly: A Case  Study on Drug Distribution to Each Tissue. 2024",
    },
    {
        "source_id": "KPD04",
        "citation": "Chen X, Zhu P, Liu B, Wei L, Xu Y (2018) Simultaneous determination of fourteen compounds of Hedyotis diffusa Willd extract in rats by UHPLC-MS/MS method: application to pharmacokinetics and tissue distribution study. J Pharm Biomed Anal 159:490-512.",
        "doi_or_url": "https://doi.org/10.1016/j.jpba.2018.07.023",
        "source_type": "journal article",
        "workbook_key": "Simultaneous determination of fourteen compounds of Hedyotis diffusa Willd extract in rats by UHPLC-MS/MS method: Application to pharmacokinetics and tissue distribution study",
    },
    {
        "source_id": "KPD05",
        "citation": "Nigade PB, Gundu J, Sreedhara Pai K, Nemmani KVS (2017) Prediction of tissue-to-plasma ratios of basic compounds in mice. Eur J Drug Metab Pharmacokinet 42:835-847.",
        "doi_or_url": "https://doi.org/10.1007/s13318-017-0402-5",
        "source_type": "journal article",
        "workbook_key": "Prediction of Tissue-to-Plasma Ratios of Basic Compounds in Mice",
    },
    {
        "source_id": "KPD06",
        "citation": "Ishida N, Kondo Y, Chikano Y, Kobayashi-Nakade E, Suga Y, Ishizaki J, Komai K, Matsushita R (2019) Pharmacokinetics and tissue distribution of 3,4-diaminopyridine in rats. Biopharm Drug Dispos 40:294-301.",
        "doi_or_url": "https://doi.org/10.1002/bdd.2203",
        "source_type": "journal article",
        "workbook_key": "Pharmacokinetics and tissue distribution of 3,4 diaminopyridine in rats",
    },
    {
        "source_id": "KPD07",
        "citation": "Bernareggi A, Rowland M (1991) Physiologic modeling of cyclosporin kinetics in rat and man. J Pharmacokinet Biopharm 19:21-50.",
        "doi_or_url": "https://doi.org/10.1007/BF01062191",
        "source_type": "journal article",
        "workbook_key": "Bernareggi A, Rowland M. Physiologic modeling of cyclosporin kinetics in rat and man. J Pharmacokinet Biopharm. 1991;19(1):21-50.",
    },
    {
        "source_id": "KPD08",
        "citation": "Yata N, Toyoda T, Murakami T, Nishiura A, Higashi Y (1990) Phosphatidylserine as a determinant for the tissue distribution of weakly basic drugs in rats. Pharm Res 7:1019-1025.",
        "doi_or_url": "https://doi.org/10.1023/A:1015935031933",
        "source_type": "journal article",
        "workbook_key": "Yata N, Toyoda T, Murakami T, Nishiura A, Higashi Y. Phosphatidylserine as a determinant for the tissue distribution of weakly basic drugs in rats. Pharm Res. 1990;7(10):1019-25.",
    },
    {
        "source_id": "KPD09",
        "citation": "Yasoshima K, Kuwabara T, Fuse E, Kuramitu T, Kurata N, Nishiie H, Oishi T, Kobayashi H, Kobayashi S (2001) Pharmacokinetics, distribution, metabolism and excretion of [3H]UCN-01 in rats and dogs after intravenous administration. Cancer Chemother Pharmacol 47:106-112.",
        "doi_or_url": "https://doi.org/10.1007/s002800000213",
        "source_type": "journal article",
        "workbook_key": "Pharmacokinetics, distribution, metabolism and excretion of [3H]UCN-01 in rats and dogs after intravenous administration",
    },
    {
        "source_id": "KPD10",
        "citation": "Paix\u00e3o P, Gouveia LF, Morais JAG (2009) Prediction of drug distribution within blood. Eur J Pharm Sci 36:544-554.",
        "doi_or_url": "https://doi.org/10.1016/j.ejps.2008.12.011",
        "source_type": "journal article",
        "workbook_key": "Prediction of drug distribution within blood",
    },
    {
        "source_id": "KPD11",
        "citation": "Shibata N, Gao W, Okamoto H, Kishida T, Iwasaki K, Yoshikawa Y, Takada K (2002) Drug interactions between HIV protease inhibitors based on physiologically-based pharmacokinetic model. J Pharm Sci 91:680-689.",
        "doi_or_url": "https://doi.org/10.1002/jps.10051",
        "source_type": "journal article",
        "workbook_key": "Drug interactions between HIV protease inhibitors based on physiologically-based pharmacokinetic model",
    },
    {
        "source_id": "KPD12",
        "citation": "Hosseini-Yeganeh M, McLachlan AJ (2001) Tissue distribution of terbinafine in rats. J Pharm Sci 90:1817-1828.",
        "doi_or_url": "https://doi.org/10.1002/jps.1132",
        "source_type": "journal article",
        "workbook_key": "Tissue distribution of terbinafine in rats",
    },
    {
        "source_id": "KPD13",
        "citation": "Kitani M, Ozaki Y, Katayama K, Kakemi M, Koizumi T (1988) A kinetic study on drug distribution: furosemide in rats. Chem Pharm Bull 36:1053-1062.",
        "doi_or_url": "https://doi.org/10.1248/cpb.36.1053",
        "source_type": "journal article",
        "workbook_key": "Kitani M, Ozaki Y, Kitayama K, Kakemi M, Koizumi T. A kinetic study on drug distribution: furosemide in rats. Chem Pharm Bull. 1988;36(3):1053-62.",
    },
    {
        "source_id": "KPD14",
        "citation": "Tsuji A, Yoshikawa T, Nishide K, Minami H, Kimura M, Nakashima E, Terasaki T, Miyamoto E, Nightingale CH, Yamana T (1983) Physiologically based pharmacokinetic model for \u03b2-lactam antibiotics I: tissue distribution and elimination in rats. J Pharm Sci 72:1239-1252.",
        "doi_or_url": "https://doi.org/10.1002/jps.2600721103",
        "source_type": "journal article",
        "workbook_key": "Tsuji A, Yoshikawa T, Nishide K, Minami H, Kimura M, Nakashima E, et al. Physiologically based pharmacokinetic model for beta-lactam antibiotics I: Tissue distribution and elimination in rats. J Pharm Sci. 1983;72(11):1239-52.",
    },
    {
        "source_id": "KPD15",
        "citation": "Yokooji T, Mori N, Murakami T (2011) Modulated pharmacokinetics and increased small intestinal toxicity of methotrexate in bilirubin-treated rats. J Pharm Pharmacol 63:206-213.",
        "doi_or_url": "https://doi.org/10.1111/j.2042-7158.2010.01213.x",
        "source_type": "journal article",
        "workbook_key": "Yokooji T, Mori N, Murakami T. Modulated pharmacokinetics and increased small intestinal toxicity of methotrexate in bilirubin-treated rats. J Pharm Pharmacol. 2011;63(2):206-13.",
    },
    {
        "source_id": "KPD16",
        "citation": "C\u00e1rcel-Trullols J, Torres-Molina F, Araico A, Saadeddin A, Peris JE (2004) Effect of cyclosporine A on the tissue distribution and pharmacokinetics of etoposide. Cancer Chemother Pharmacol 54:153-160.",
        "doi_or_url": "https://doi.org/10.1007/s00280-004-0784-3",
        "source_type": "journal article",
        "workbook_key": "Carcel-Trullols J, Torres-Molina F, Araico A, Saadeddin A, Peris JE. Effect of cyclosporine A on the tissue distribution and pharmacokinetics of etoposide. Cancer Chemother Pharmacol. 2004;54(2):153-60.",
    },
    {
        "source_id": "KPD17",
        "citation": "Song D, Sun L, DuBois DC, Almon RR, Meng S, Jusko WJ (2020) Physiologically based pharmacokinetics of dexamethasone in rats. Drug Metab Dispos 48:811-818.",
        "doi_or_url": "https://doi.org/10.1124/dmd.120.091017",
        "source_type": "journal article",
        "workbook_key": "Song D, Sun L, DuBois DC, Almon RR, Meng S, Jusko WJ. Physiologically based pharmacokinetics of dexamethasone in rats. Drug Metab Dispos. 2020;48(9):811-8.",
    },
    {
        "source_id": "KPD18",
        "citation": "Li Z, Gao Y, Yang C, Xiang Y, Zhang W, Zhang T, Su R, Lu C, Zhuang X (2020) Assessment and confirmation of species difference in nonlinear pharmacokinetics of atipamezole with physiologically based pharmacokinetic modeling. Drug Metab Dispos 48:41-51.",
        "doi_or_url": "https://doi.org/10.1124/dmd.119.089151",
        "source_type": "journal article",
        "workbook_key": "Assessment and Confirmation of Species Difference in Nonlinear Pharmacokinetics of Atipamezole with Physiologically Based Pharmacokinetic Modeling",
    },
    {
        "source_id": "KPD19",
        "citation": "DrugBank (2026) DrugBank Online. OMx Personal Health Analytics, Edmonton. Accessed 4 September 2026.",
        "doi_or_url": "https://go.drugbank.com/",
        "source_type": "online database",
        "workbook_key": "drugbank",
    },
    {
        "source_id": "KPD20",
        "citation": "Yun YE (2013) Development of a correlation based and a decision tree based prediction algorithm for tissue to plasma partition coefficients (Kp). Dissertation, University of Waterloo.",
        "doi_or_url": "",
        "source_type": "doctoral dissertation",
        "workbook_key": "Development of a correlation based and a decision tree based prediction algorithm for tissue to plasma partition coefficients. 2013",
    },
]


def read_first_sheet(path: Path) -> list[list[object]]:
    """Read an .xlsx first worksheet with only the standard library."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append("".join(node.text or "" for node in item.iterfind(".//a:t", ns)))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[list[object]] = []
    for row in sheet.findall(".//a:sheetData/a:row", ns):
        values: dict[int, object] = {}
        for cell in row.findall("a:c", ns):
            ref = cell.attrib["r"]
            letters = "".join(ch for ch in ref if ch.isalpha())
            col = 0
            for ch in letters:
                col = col * 26 + ord(ch.upper()) - 64
            kind = cell.attrib.get("t")
            value_node = cell.find("a:v", ns)
            if value_node is None:
                value: object = ""
            elif kind == "s":
                value = shared[int(value_node.text)]
            else:
                value = value_node.text or ""
            values[col - 1] = value
        width = max(values, default=-1) + 1
        rows.append([values.get(index, "") for index in range(width)])
    return rows


def build(raw_csv: Path, reference_workbook: Path) -> pd.DataFrame:
    raw = prepare_data.read_csv(raw_csv)
    workbook_rows = read_first_sheet(reference_workbook)
    header = workbook_rows[0]
    reference_index = header.index("Reference")
    records = workbook_rows[1:]
    if len(records) != len(raw):
        raise RuntimeError(f"Source-row mismatch: workbook={len(records)}, CSV={len(raw)}")
    workbook_keys = [str(row[reference_index]) if reference_index < len(row) else "" for row in records]
    if list(raw["Drug"].fillna("").astype(str)) != [str(row[0]) for row in records]:
        raise RuntimeError("Workbook and CSV drug-label order does not match")

    manifest = prepare_data.add_identity_ids(prepare_data.build_resolution_manifest(raw))
    manifest["workbook_key"] = workbook_keys
    condition = prepare_data.condition_specific_long(manifest)
    primary = prepare_data.aggregate_primary(condition)
    if len(primary) != 1270:
        raise RuntimeError(f"Expected 1,270 primary records, found {len(primary)}")

    def normalize_key(value: object) -> str:
        text = str(value).replace("\u2013", "-").replace("\u2014", "-").replace("\u03b2", "beta")
        return re.sub(r"\s+", " ", text).strip().lower()

    source_id_by_key = {normalize_key(item["workbook_key"]): item["source_id"] for item in SOURCES}
    unknown = sorted({key for key in workbook_keys if normalize_key(key) not in source_id_by_key})
    if unknown:
        raise RuntimeError(f"Unmapped workbook references: {unknown}")
    manifest["source_id"] = manifest["workbook_key"].map(lambda value: source_id_by_key[normalize_key(value)])
    source_id_by_row_id = manifest.set_index("source_row_id")["source_id"].to_dict()

    primary_contributions: dict[str, int] = {item["source_id"]: 0 for item in SOURCES}
    for row_ids in primary["source_row_ids"].astype(str):
        represented = {source_id_by_row_id[row_id] for row_id in row_ids.split(" | ")}
        for source_id in represented:
            primary_contributions[source_id] += 1

    condition_counts = condition.groupby("source_row_id").size().to_dict()
    manifest["positive_condition_level_measurements"] = manifest["source_row_id"].map(condition_counts).fillna(0).astype(int)

    rows = []
    for item in SOURCES:
        source = manifest[manifest["source_id"].eq(item["source_id"])]
        species = "; ".join(sorted(source["species_normalized"].dropna().astype(str).unique()))
        retained = source[source["included_in_primary"].astype(bool)]
        n_primary = primary_contributions[item["source_id"]]
        if n_primary:
            role = "contributed to the 1,270-record rat primary analysis"
        elif len(source) == 0:
            role = "listed in the supplied bibliography but not linked to a row in the source workbook"
        elif not retained.empty:
            role = "retained source row(s) produced no eligible primary record"
        else:
            role = "raw collection only; did not contribute to the rat primary analysis"
        rows.append(
            {
                "source_id": item["source_id"],
                "source_type": item["source_type"],
                "citation": item["citation"],
                "doi_or_url": item["doi_or_url"],
                "species_present_in_source_workbook": species or "not linked",
                "raw_collection_rows": int(len(source)),
                "retained_rat_source_rows": int(len(retained)),
                "positive_condition_level_measurements": int(source["positive_condition_level_measurements"].sum()),
                "primary_aggregated_records_contributed": int(n_primary),
                "provenance_role": role,
            }
        )
    result = pd.DataFrame(rows)
    if int(result["positive_condition_level_measurements"].sum()) != len(condition):
        raise RuntimeError("Condition-level contribution counts do not reconcile")
    return result


def write_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "kp_data_source_bibliography.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    contributing = frame[frame["primary_aggregated_records_contributed"].gt(0)]
    lines = [
        "# Kp_Data source bibliography and aggregate provenance",
        "",
        "This file documents sources associated with the internally curated, non-public Kp_Data collection. It does not redistribute compound names, structures, tissue-level Kp values, or record-level source links.",
        "",
        "The source workbook was row-aligned to Kp_Data and processed with the same frozen curation and aggregation code used for the manuscript. The resulting primary dataset contained 1,270 positive rat compound-tissue records. A source is marked as contributing when at least one retained source row entered a final aggregated record. Because multiple source rows can be aggregated into one compound-tissue record, per-source contribution counts are not expected to sum to 1,270.",
        "",
        f"Of the {len(frame)} supplied bibliography entries, {len(contributing)} contributed to the final rat primary analysis.",
        "",
        "| ID | Source | DOI/URL | Workbook species | Primary records contributed | Role |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        citation = str(row.citation).replace("|", "\\|")
        role = str(row.provenance_role).replace("|", "\\|")
        lines.append(
            f"| {row.source_id} | {citation} | {row.doi_or_url} | "
            f"{row.species_present_in_source_workbook} | {row.primary_aggregated_records_contributed} | {role} |"
        )
    lines.extend(
        [
            "",
            "## Count definitions",
            "",
            "- `raw_collection_rows`: wide-format rows linked to the bibliography entry before species and curation filtering.",
            "- `retained_rat_source_rows`: rat rows retained by the frozen identity and target-availability rules before tissue-level melting.",
            "- `positive_condition_level_measurements`: eligible positive tissue measurements before aggregation across experimental conditions.",
            "- `primary_aggregated_records_contributed`: final compound-tissue records to which the source contributed at least one retained measurement.",
            "",
            "Bibliographic metadata were supplied by the authors and should be checked against the cited originals before manuscript submission.",
        ]
    )
    (output_dir / "kp_data_source_bibliography.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--reference-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    frame = build(args.raw_csv, args.reference_workbook)
    write_outputs(frame, args.output_dir)
    print(frame[["source_id", "raw_collection_rows", "retained_rat_source_rows", "positive_condition_level_measurements", "primary_aggregated_records_contributed"]].to_string(index=False))


if __name__ == "__main__":
    main()
