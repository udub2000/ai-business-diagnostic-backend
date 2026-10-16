"""Read-only, non-generative document evidence preview. No scoring changes."""
from __future__ import annotations
import csv
from io import StringIO
from pathlib import Path
import re
from typing import Any
from .geometry_v45 import read as read_geometry

SUPPORTED = {'.csv', '.xlsx', '.xls', '.pdf', '.docx'}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_FILES = 5
MAX_EXCERPTS = 18
MAX_TEXT = 220
PATTERNS = {
    'Revenue / sales': r'\b(revenue|sales|turnover|income from operations)\b',
    'Costs / expenses': r'\b(expense|cost of (goods|sales|services)|payroll|labor cost|operating costs?)\b',
    'Profitability': r'\b(gross profit|net profit|net income|margin|ebitda|operating profit)\b',
    'Cash / receivables': r'\b(cash flow|accounts receivable|receivables|outstanding invoices|cash balance)\b',
    'Demand / conversion': r'\b(leads?|conversion|pipeline|inquiries|bookings|quotes?)\b',
    'Capacity / operations': r'\b(utilization|capacity|backlog|crew|jobs completed|scheduling|cycle time|turnaround)\b',
    'Customer / retention': r'\b(retention|repeat customers?|complaints?|refunds?|churn)\b',
}


def _short(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()[:MAX_TEXT]


def _records(path: Path):
    ext = path.suffix.lower()
    if ext in {'.xlsx', '.docx'}:
        data = read_geometry(path)
        if ext == '.xlsx':
            for sheet in data['sheets']:
                for row in sheet['rows']:
                    cells = row['cells']
                    for c in cells:
                        if c.get('value') not in (None, ''):
                            yield f"{sheet['name']}!{c['address']}", _short(c['value'])
        else:
            for index, block in enumerate(data['blocks'], 1):
                if block['kind'] == 'paragraph':
                    yield f'paragraph {index}', _short(block['text'])
                else:
                    for row in block['rows']:
                        for c in row['cells']:
                            yield c['locator'], _short(c['value'])
    elif ext == '.csv':
        raw = path.read_bytes()[:MAX_FILE_BYTES]
        content = raw.decode('utf-8-sig', errors='replace')
        for row_no, row in enumerate(csv.reader(StringIO(content)), 1):
            for col_no, value in enumerate(row, 1):
                yield f'row {row_no}, column {col_no}', _short(value)
    elif ext == '.xls':
        import xlrd
        wb = xlrd.open_workbook(str(path), on_demand=True)
        try:
            for sheet in wb.sheets():
                for ri in range(sheet.nrows):
                    for ci in range(sheet.ncols):
                        value = sheet.cell_value(ri, ci)
                        if value not in ('', None):
                            yield f'{sheet.name}!R{ri+1}C{ci+1}', _short(value)
        finally:
            wb.release_resources()
    elif ext == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for page_no, page in enumerate(reader.pages[:40], 1):
            text = page.extract_text() or ''
            for line_no, line in enumerate(text.splitlines()[:2000], 1):
                yield f'page {page_no}, line {line_no}', _short(line)
    else:
        raise ValueError('Unsupported format')


def preview_document(path: Path, display_name: str) -> dict[str, Any]:
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError('Unsupported file type')
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('File exceeds the 8 MB preview limit')
    found: list[dict[str, str]] = []
    categories: set[str] = set()
    scanned = 0
    truncated = False
    for locator, value in _records(path):
        scanned += 1
        if scanned > 100_000:
            truncated = True
            break
        if not value:
            continue
        for category, regex in PATTERNS.items():
            if re.search(regex, value, flags=re.IGNORECASE):
                categories.add(category)
                if len(found) < MAX_EXCERPTS:
                    found.append({'category': category, 'locator': locator, 'excerpt': value})
    limitations = [
        'Keyword-based candidates only; no verified business metrics or conclusions.',
        'Values, periods, units, and relationships have not been independently verified.',
        'Document evidence does not affect diagnostic scores or recommendations.',
    ]
    if path.suffix.lower() == '.pdf':
        limitations.append('Scanned/image-only PDF pages may have no extractable text.')
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        limitations.append('Formula calculations and workbook relationships are not validated.')
    if truncated:
        limitations.append('Scanning stopped at the 100,000-record safety limit.')
    return {'file_name': display_name, 'format': path.suffix.lower().lstrip('.'),
            'records_scanned': min(scanned, 100_000), 'categories_detected': sorted(categories),
            'evidence_candidates': found, 'limitations': limitations,
            'status': 'review_required', 'read_only': True}
