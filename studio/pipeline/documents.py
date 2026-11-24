"""Read normal Word scripts, preserving automatic list numbers and table order."""
from io import BytesIO
import re
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table


def word_script_text(data):
    document=Document(BytesIO(data));numbers={};lines=[]
    # Documents with literal step numbers may not contain a Word numbering part.
    # Python-docx raises NotImplementedError when that relationship is absent.
    # These documents must remain valid inputs; don't require a numbering part
    # unless an actual Word list needs number reconstruction.
    try:
        numbering=document.part.numbering_part.element
    except (KeyError,NotImplementedError):
        numbering=None

    def paragraph_text(p):
        text=p.text.strip()
        if not text:return ''
        text=re.sub(r'^Step\s+(\d+)\s*[:.)-]\s*',r'\1. ',text,flags=re.I)
        if re.match(r'^\d+\s*[.)-]\s*',text):return text
        numpr=p._p.pPr.numPr if p._p.pPr is not None else None
        style=p.style;seen=set()
        while numpr is None and style is not None and style.style_id not in seen:
            seen.add(style.style_id);pr=style.element.pPr
            if pr is not None:numpr=pr.numPr
            style=style.base_style
        if numpr is None or numpr.numId is None or numbering is None:return text
        ident=str(numpr.numId.val);level=int(numpr.ilvl.val) if numpr.ilvl is not None else 0
        num=next((n for n in numbering.findall(qn('w:num')) if n.get(qn('w:numId'))==ident),None)
        if num is None:return text
        abstract_id=num.find(qn('w:abstractNumId')).get(qn('w:val'))
        abstract=next((n for n in numbering.findall(qn('w:abstractNum')) if n.get(qn('w:abstractNumId'))==abstract_id),None)
        lvl=next((n for n in abstract.findall(qn('w:lvl')) if n.get(qn('w:ilvl'))==str(level)),None) if abstract is not None else None
        if lvl is None:return text
        fmt=lvl.find(qn('w:numFmt'))
        if fmt is None or fmt.get(qn('w:val')) not in {'decimal','decimalZero'}:return text
        start=lvl.find(qn('w:start'));initial=int(start.get(qn('w:val'))) if start is not None else 1
        for override in num.findall(qn('w:lvlOverride')):
            if override.get(qn('w:ilvl'))==str(level):
                value=override.find(qn('w:startOverride'))
                if value is not None:initial=int(value.get(qn('w:val')))
        key=(ident,level);numbers[key]=numbers.get(key,initial-1)+1
        # Nested items stay associated with the preceding protocol step.
        prefix=str(numbers.get((ident,0),1))+'.'+str(numbers[key]) if level else str(numbers[key])
        return prefix+'. '+text

    for element in document.element.body:
        if element.tag==qn('w:p'):
            text=paragraph_text(Paragraph(element,document))
            if text:lines.append(text)
        elif element.tag==qn('w:tbl'):
            for row in Table(element,document).rows:
                seen=set()
                for cell in row.cells:
                    if cell._tc in seen:continue
                    seen.add(cell._tc)
                    lines.extend(text for p in cell.paragraphs if (text:=paragraph_text(p)))
    return '\n'.join(lines)
