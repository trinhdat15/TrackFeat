#!/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python
from pathlib import Path
import hashlib,json,re,subprocess
from datetime import datetime,timezone
import pandas as pd
ROOT=Path('/ssd1/team_cam_ai/ttdat'); OUT=ROOT/'final_draft/trackfeat_aisi_submission_01'; FIG=OUT/'figures'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def cmd(*x):return subprocess.check_output(x,text=True,stderr=subprocess.STDOUT)
required=['upstream_perception_contract.csv','upstream_selection_evidence.md','full_132_feature_contract.csv','reduced_52_feature_contract.csv','feature_contract_comparison.csv','feature_selection_lineage.md','paired_validation_bootstrap.csv','calibration_policy_registry.csv','calibration_pareto_frontier.csv','paper_statistic_ledger.csv','paper_claim_evidence_map.csv','timestamp_diagnostic_summary.md','paper_evidence_manifest.json','figure_pipeline.pdf','figure_feature_family_heatmap.pdf','figure_feature_level_heatmap_supplement.pdf','figure_calibration_transfer.pdf','main.tex','references.bib','result_macros.tex','aaai2027.sty','aaai2027.bst','main.pdf']
missing=[x for x in required if not (OUT/x).exists()]
log=(OUT/'main.log').read_text(errors='replace'); tex=(OUT/'main.tex').read_text(errors='replace')
info=cmd('pdfinfo',str(OUT/'main.pdf')); pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1)); size=re.search(r'^Page size:\s+(.+)',info,re.M).group(1)
fonts=cmd('pdffonts',str(OUT/'main.pdf')); flines=[x.split() for x in fonts.splitlines()[2:] if x.strip()]; embedded=all(('yes' in [z.lower() for z in x]) for x in flines); type3=any('Type 3' in x for x in fonts.splitlines())
placeholder_terms=re.findall(r'(?i)\b(?:TODO|TBD|PLACEHOLDER|FIXME)\b|\?\?',tex)
# Rendered-page structural inventory and text questions.
aud=OUT/'audit_pages'; aud.mkdir(exist_ok=True)
subprocess.run(['pdftoppm','-png','-r','120',str(OUT/'main.pdf'),str(aud/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
page_rows=[]
for n in range(1,pages+1):
 text=cmd('pdftotext','-f',str(n),'-l',str(n),'-layout',str(OUT/'main.pdf'),'-')
 page_rows.append({'page':n,'role':['abstract_and_introduction','related_work','method','experimental_setup','references'][n-1] if n<=5 else 'unexpected','text_characters':len(text),'figure_1_present':'Fixed perception, explicit features' in text,'figure_2_present':'Training-OOF family allocation' in text,'table_1_present':'Frozen SO-TAD roles' in text,'references_present':'References' in text,'render_path':str((aud/f'page-{n}.png').relative_to(OUT))})
pd.DataFrame(page_rows).to_csv(OUT/'page_structure_audit.csv',index=False)
audit={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PAGES_1_TO_4_COMPILED_AND_AUDITED','page_count_total':pages,'technical_page_count':4,'reference_page_count':pages-4,'page_size':size,'letter_size':('612 x 792' in size),'overfull_box_count':len(re.findall(r'Overfull \\hbox|Overfull \\vbox',log)),'unresolved_reference_count':len(re.findall(r'undefined references|Reference .* undefined',log,re.I)),'unresolved_citation_count':len(re.findall(r'undefined citations|Citation .* undefined',log,re.I)),'missing_figure_count':len(re.findall(r'File .* not found|pdftex.def Error',log,re.I)),'placeholder_count':len(placeholder_terms),'fonts_embedded':embedded,'type3_fonts':type3,'required_artifacts_missing':missing,'figure_1_page':next((r['page'] for r in page_rows if r['figure_1_present']),None),'figure_2_page':next((r['page'] for r in page_rows if r['figure_2_present']),None),'table_1_page':next((r['page'] for r in page_rows if r['table_1_present']),None),'references_first_page':next((r['page'] for r in page_rows if r['references_present']),None),'validation_used':'frozen decisions only; no rescoring or selection','timestamp_tsc_validation_evaluated':False,'official_test_accessed':False,'main_pdf_sha256':sha(OUT/'main.pdf'),'main_tex_sha256':sha(OUT/'main.tex'),'font_report':fonts}
(OUT/'paper_compile_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
# Refresh the evidence manifest after manuscript and final figure sealing.
artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.name not in {'paper_evidence_manifest.json'} and 'audit_pages/' not in str(p.relative_to(OUT)) and p.suffix not in {'.aux','.log','.out','.blg','.bbl'}]
manifest={'sealed_utc':datetime.now(timezone.utc).isoformat(),'scope':'frozen evidence extraction and manuscript Pages 1-4 only','scientific_experiment_run':False,'model_training_or_tuning':False,'validation_prediction_rescoring':False,'timestamp_tsc_validation_evaluated':False,'official_test_accessed':False,'feature_contract_counts':{'full':132,'reduced':52},'validation_population':{'total':332,'accident':46,'normal':286},'bootstrap':{'replicates':2000,'unit':'paired video','stratified':True,'seed':130057},'calibration_policy_count':356,'training_oof_attribution_videos':1482,'artifact_hashes':{str(p.relative_to(OUT)):sha(p) for p in sorted(artifacts)},'compile_audit':audit}
(OUT/'paper_evidence_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
# Self-hash receipt avoids recursive inclusion in the manifest.
receipt={'paper_evidence_manifest_sha256':sha(OUT/'paper_evidence_manifest.json'),'main_pdf_sha256':sha(OUT/'main.pdf'),'main_tex_sha256':sha(OUT/'main.tex'),'references_bib_sha256':sha(OUT/'references.bib')}
(OUT/'seal_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'audit':audit,'seal_receipt':receipt},indent=2))
