"""Read-only OOXML geometry extraction. No inferred headers or business semantics."""
from pathlib import Path
from xml.etree import ElementTree as ET
import zipfile,re,json
S='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
R='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
P='{http://schemas.openxmlformats.org/package/2006/relationships}'

def col_number(label):
 n=0
 for char in label.upper():
  if not 'A'<=char<='Z':raise ValueError('invalid column')
  n=n*26+ord(char)-64
 return n

def read(path):
 path=Path(path)
 if path.suffix.lower() not in ('.xlsx','.docx'):raise ValueError('Only xlsx/docx supported')
 with zipfile.ZipFile(path) as z:
  if sum(i.file_size for i in z.infolist())>30_000_000:raise ValueError('OOXML exceeds size limit')
  return _xlsx(z) if path.suffix.lower()=='.xlsx' else _docx(z)

def _xlsx(z):
 shared=[]
 if 'xl/sharedStrings.xml' in z.namelist():
  root=ET.fromstring(z.read('xl/sharedStrings.xml'))
  shared=[''.join(x.text or '' for x in si.iter(S+'t')) for si in root.findall(S+'si')]
 # Workbook order and relationship targets; no alphabetical sheet assumption.
 wb=ET.fromstring(z.read('xl/workbook.xml'))
 relroot=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
 rels={r.get('Id'):r.get('Target') for r in relroot.findall(P+'Relationship')}
 sheets=[]
 for sheet in wb.findall('.//'+S+'sheets/'+S+'sheet'):
  target=rels[sheet.get(R+'id')]
  part=target.lstrip('/') if target.startswith('/') else 'xl/'+target
  part=re.sub(r'^xl/xl/','xl/',part)
  root=ET.fromstring(z.read(part)); rows=[]
  for row in root.findall('.//'+S+'sheetData/'+S+'row'):
   cells=[]
   for c in row.findall(S+'c'):
    ref=c.get('r');m=re.fullmatch(r'([A-Z]+)(\d+)',ref or '')
    if not m:continue
    v=c.find(S+'v');inline=c.find(S+'is');formula=c.find(S+'f')
    if v is not None:
     val=v.text or ''
     if c.get('t')=='s':val=shared[int(val)]
    elif inline is not None:val=''.join(t.text or '' for t in inline.iter(S+'t'))
    else:val=None
    cells.append({'address':ref,'column':col_number(m.group(1)),'row':int(m.group(2)),'value':val,'style':c.get('s','0'),'formula':formula.text if formula is not None else None,'present':True})
   rows.append({'row':int(row.get('r')),'cells':cells})
  merged=[m.get('ref') for m in root.findall('.//'+S+'mergeCells/'+S+'mergeCell')]
  sheets.append({'name':sheet.get('name'),'part':part,'rows':rows,'merged_ranges':merged})
 return {'format':'xlsx','sheets':sheets,'review_only':True,'geometry_verified':True}

def _docx(z):
 root=ET.fromstring(z.read('word/document.xml'));body=root.find(W+'body');blocks=[];table_index=0
 for block in body:
  if block.tag==W+'p':
   style=block.find(W+'pPr/'+W+'pStyle')
   blocks.append({'kind':'paragraph','text':''.join(t.text or '' for t in block.iter(W+'t')),'style':style.get(W+'val') if style is not None else None})
  elif block.tag==W+'tbl':
   table_index+=1;rows=[]
   grid=block.find(W+'tblGrid');columns=len(grid.findall(W+'gridCol')) if grid is not None else None
   for ri,tr in enumerate(block.findall(W+'tr'),1):
    props=tr.find(W+'trPr');before=props.find(W+'gridBefore') if props is not None else None
    cursor=int(before.get(W+'val','0')) if before is not None else 0
    marker=props.find(W+'tblHeader') if props is not None else None
    explicit=marker is not None and marker.get(W+'val','1') not in ('0','false','off')
    cells=[]
    for ci,tc in enumerate(tr.findall(W+'tc'),1):
     pr=tc.find(W+'tcPr');span=pr.find(W+'gridSpan') if pr is not None else None
     width=max(1,int(span.get(W+'val','1'))) if span is not None else 1
     merge=pr.find(W+'vMerge') if pr is not None else None
     cells.append({'locator':f'table[{table_index}]/row[{ri}]/cell[{ci}]','column_start':cursor+1,'column_span':width,'value':''.join(t.text or '' for t in tc.iter(W+'t')),'vertical_merge':merge.get(W+'val','continue') if merge is not None else None})
     cursor+=width
    rows.append({'row':ri,'cells':cells,'explicit_header':explicit,'grid_width':cursor})
   blocks.append({'kind':'table','table_index':table_index,'grid_columns':columns,'rows':rows})
 return {'format':'docx','blocks':blocks,'review_only':True,'geometry_verified':True}

def dense_excel_row(row):
 """Present original positions with None for absent coordinates, not empty strings."""
 cells=row['cells']
 if not cells:return []
 columns={c['column']:c['value'] for c in cells}
 return [columns.get(i) for i in range(1,max(columns)+1)]
