ZeBRA Model for SISA Screening

You can run ZeBRA using API call

# `zebra-predict-sisa`


### Single Patient Example

Specify `target` (`M2550`, `F2550`, `M5075`, or `F5075`) to run ZeBRA for different patient cohorts.

```
!curl -X POST -H "Content-Type: application/json" -d @SAMPLE_INPUT/M2550.json "https://us-central1-pkcsaas-01.cloudfunctions.net/zebra-predict-sisa?target=M2550&api_key=63d87942b4a2d61985a377db05a35e73"
```

## Input Format

Input is provided as a JSON file containing a list of patient records. Each patient record is represented as a JSON object with the fields below.

- `patient_id` - numeric value, can be deidentified ID.
- `birth_date` - Format: "MM-DD-YYYY". Does not have to be exact date of birth, but has to be approximate enough to estimate patient's age.
- `sex` - "M" and "F" values are accepted.
- `DX_record` - A list of dictionaries, each containing a date of diagnosis in "MM-DD-YYYY" format (`date`), and a diagnostic code in ICD-10 or ICD-9 format (`code`).
- `RX_record` - A list of dictionaries, each containing a date of prescription in "MM-DD-YYYY" format (`date`), and a prescription code in [NDC (National Drug Code) format](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory) (`code`).
- `PROC_record` - A list of dictionaries, each containing a date of procedure in "MM-DD-YYYY" format (`date`), and a procedural code in CPT, HCPCS, or ICD format (`code`).

### Example Input (valid JSON)   

```json
[
    { 
        'patient_id': 33443873802,
        'birth_date': '01-01-2006',
        'sex': 'F',
        'DX_record': [
            {'date': '07-31-2006', 'code': 'Z38.00'},
            {'date': '08-07-2006', 'code': 'Z00.129'},
            {'date': '08-07-2006', 'code': 'P59.9'},
            {'date': '08-29-2016', 'code': 'J01.90'}
        ],
        'RX_record': [
            {'date': '10-29-2011', 'code': '61168010101'},
            {'date': '05-16-2015', 'code': '00071439703'},
            {'date': '08-08-2015', 'code': '00005501523'},
        ],
        'PROC_record': [
            {'date': '02-05-2007', 'code': '90723'},
            {'date': '11-05-2007', 'code': 'J1100'},
            {'date': '11-05-2007', 'code': '99214'},
        ]
    }
]
```
 
### Example output for one patient:

```json
[
    {
        'patient_id': 33443873802,
        'predicted_risk': 0.00353191883909529,
        'error_code': '',
        'probability': 0.9696941114273572
    }
]
```

## Input Data Requirements
- Each record must span at least 1 year between the earliest and latest diagnosis dates in `DX_record`. Records with shorter diagnostic histories are not assessed in order to ensure sufficient longitudinal context for feature construction.
