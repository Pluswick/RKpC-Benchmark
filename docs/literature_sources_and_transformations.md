# PT and RR source locations and transformation rules

No source-paper record values are included in this repository.

## Poulin-Theil benchmark

Source: Poulin P, Theil F-P. A priori prediction of tissue:plasma partition coefficients of drugs to facilitate the use of physiologically-based pharmacokinetic models in drug discovery. *Journal of Pharmaceutical Sciences*. 2000;89:16-35. DOI: 10.1002/(SICI)1520-6017(200001)89:1<16::AID-JPS3>3.0.CO;2-E.

Predicted and experimental Kp entries were transcribed from Table III. Single reported experimental values and reported means were retained. A hyphenated two-study range was represented by its arithmetic midpoint. The original transcription, both range endpoints, and the derived midpoint remain in the internal provenance record and are not redistributed.

## Rodgers-Rowland benchmark

Sources:

1. Rodgers T, Leahy D, Rowland M. Physiologically based pharmacokinetic modeling 1: Predicting the tissue distribution of moderate-to-strong bases. *Journal of Pharmaceutical Sciences*. 2005;94:1259-1276. DOI: 10.1002/jps.20322.
2. Rodgers T, Rowland M. Physiologically based pharmacokinetic modelling 2: Predicting the tissue distribution of acids, very weak bases, neutrals and zwitterions. *Journal of Pharmaceutical Sciences*. 2006;95:1238-1257. DOI: 10.1002/jps.20502.
3. Rodgers T, Leahy D, Rowland M. Erratum. *Journal of Pharmaceutical Sciences*. 2007;96:3151-3152. DOI: 10.1002/jps.20856.
4. Rodgers T, Rowland M. Erratum. *Journal of Pharmaceutical Sciences*. 2007;96:3153-3154. DOI: 10.1002/jps.20857.

Source-reported tissue-to-unbound-plasma-water coefficients were converted to tissue-to-plasma Kp as `Kp = fu * Kpu`, using the corresponding source-paper fraction unbound in plasma. Source-exact values from the 2007 errata were applied only in the derived prediction column. Where an erratum stated a change of no more than 10% without reporting an exact replacement, the published table value was retained. The pipemidic-acid adipose entry was checked against Table 5 of the 2006 article. The penicillin and pentazocine experimental blocks were corrected as a complete transposition only in the derived analysis copy.

## Matched direct panels

PT and RR records were never pooled. Each paper benchmark was linked separately to pre-existing held-out GNN predictions by exact RDKit canonical SMILES and tissue. Only predictions from splits in which the linked parent or scaffold group was held out were eligible. The optional public evaluator requires authorised, locally reconstructed record tables and writes only local record-level outputs.
