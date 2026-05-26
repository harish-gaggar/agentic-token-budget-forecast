# IEEE TAI journal submission package

Build all required Word files:

```bash
python submission/ieee-tai/build_submission.py
```

Outputs land in **`submission/ieee-tai/output/`**.

| Portal field | Upload this file |
|--------------|------------------|
| Main Manuscript (anonymized) | **`Anonymized_Main_Manuscript_LaTeX.zip`** (recommended) or `Anonymized_Main_Manuscript.docx` |
| Title Page | `Title_Page.docx` |
| Conflict of Interest | `Conflict_of_Interest.docx` |
| Cover letter (optional, editor only) | `Cover_Letter.docx` |
| Supplementary Material for Review | `Supplementary_Material_for_Review.docx` and `Anonymized_Supplementary_LaTeX.zip` |

Read **`output/SUBMISSION_CHECKLIST.md`** before uploading.

**Main manuscript:** run `make ieee-blind-pdf`, then
`python submission/ieee-tai/build_submission.py`.

The IEEE TAI portal asks for **anonymized Word or LaTeX** (not PDF).

1. **Recommended:** In the portal, choose **Anonymized Main Document - LaTeX File** and upload
   **`output/Anonymized_Main_Manuscript_LaTeX.zip`**. IEEE compiles with IEEEtran — same
   layout as **`Main_Manuscript.pdf`** (preview only).
2. **If you must use Word:** Choose **Anonymized Main Document - MS Word** and upload
   **`output/Anonymized_Main_Manuscript.docx`**. Proofread figures; it will not match the PDF
   exactly.

Official template copy (for formatting): `IEEE_TAI_Word_Template.doc`
