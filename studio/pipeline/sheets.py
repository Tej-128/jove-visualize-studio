"""Read both native-Sheets exports and Office workbooks; never overwrite references."""
from io import BytesIO
from openpyxl import load_workbook
import re

def seconds(value):
    if isinstance(value,(int,float)):return float(value)
    s=str(value).strip()
    bits=s.split(':')
    if not 1<=len(bits)<=3:raise ValueError(f'Invalid timestamp: {s}')
    total=0.
    for bit in bits:total=total*60+float(bit)
    if total<0:raise ValueError('Negative timestamp')
    return total

def read_candidates(data:bytes):
    import zipfile
    with zipfile.ZipFile(BytesIO(data)) as z:
        if sum(i.file_size for i in z.infolist())>50*1024**2:raise ValueError('Reference workbook expands beyond the pilot limit')
    book=load_workbook(BytesIO(data),read_only=True,data_only=True)
    try:
        for sheet in book:
            rows=sheet.iter_rows(values_only=True)
            header=next(rows,())
            names=[str(x or '').strip() for x in header]
            if 'Step' not in names or 'Journal ID' not in names:continue
            result=[]
            for row in rows:
                item=dict(zip(names,row))
                if item.get('Step') is None:continue
                refined=str(item.get('Exact time stamps') or '').strip()
                bounds=re.split(r'\s*[-–—]\s*',refined) if refined else []
                start,end=(seconds(x) for x in bounds) if len(bounds)==2 else (seconds(item['Start']),seconds(item['End']))
                if end<=start:raise ValueError('Candidate ends before it starts')
                result.append({'step':int(item['Step']),'text':str(item.get('Script Step') or ''),
                    'video_id':str(item['Journal ID']).removesuffix('.0'),'start':start,'end':end,
                    'feedback':str(item.get('Feedback Notes') or ''),'origin':'reference_workbook',
                    'original_start':item.get('Start'),'original_end':item.get('End')})
            return result
    finally:book.close()
    raise ValueError('No worksheet has Step and Journal ID headers.')

