from pathlib import Path
import html, math, base64, shutil, subprocess
from xml.etree.ElementTree import Element, SubElement, tostring

OUT = Path('/mnt/data/paperM_figures_v3')
SRC = OUT/'source'; MAIN=OUT/'main'; SUP=OUT/'supplementary'; PRE=OUT/'previews'
for p in [SRC, MAIN, SUP, PRE]: p.mkdir(parents=True, exist_ok=True)

# ---------- SVG helpers ----------
class SVG:
    def __init__(self, w, h):
        self.w=w; self.h=h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                    '<defs>',
                    '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#334155"/></marker>',
                    '<marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#2563eb"/></marker>',
                    '<style>text{font-family:Arial,Helvetica,sans-serif; fill:#111827}.small{font-size:18px}.tiny{font-size:15px}.label{font-size:20px;font-weight:700}.paneltitle{font-size:27px;font-weight:700}.subtitle{font-size:22px;font-weight:700}.math{font-family:Georgia,Times New Roman,serif;font-style:italic}</style>',
                    '</defs>',
                    '<rect x="0" y="0" width="100%" height="100%" fill="white"/>']
    def add(self,s): self.parts.append(s)
    def rect(self,x,y,w,h,fill='white',stroke='#cbd5e1',sw=2,rx=14, dash=None):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    def line(self,x1,y1,x2,y2,stroke='#475569',sw=2,dash=None,arrow=False):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        m=' marker-end="url(#arrow)"' if arrow else ''
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')
    def circle(self,cx,cy,r,fill='#e2e8f0',stroke='#475569',sw=2):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self,x,y,text,size=20,weight=400,anchor='start',fill='#111827',italic=False):
        esc=html.escape(text)
        style=f'font-size:{size}px;font-weight:{weight};fill:{fill};'+('font-style:italic;' if italic else '')
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{style}">{esc}</text>')
    def multiline(self,x,y,lines,size=18,weight=400,anchor='start',fill='#111827',lh=1.25):
        style=f'font-size:{size}px;font-weight:{weight};fill:{fill};'
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{style}">')
        for i,line in enumerate(lines):
            dy='0' if i==0 else str(int(size*lh))
            self.add(f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>')
        self.add('</text>')
    def polyline(self,pts,stroke='#475569',sw=2,dash=None,arrow=False,fill='none'):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        m=' marker-end="url(#arrow)"' if arrow else ''
        ps=' '.join(f'{x},{y}' for x,y in pts)
        self.add(f'<polyline points="{ps}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')
    def finish(self):
        return ''.join(self.parts)+"</svg>"

# ---------- Draw.io helpers ----------
class Drawio:
    def __init__(self, page_w, page_h):
        self.mxfile=Element('mxfile', host='app.diagrams.net', modified='2026-09-10T00:00:00.000Z', agent='OpenAI', version='24.7.17')
        self.diagram=SubElement(self.mxfile,'diagram',name='Page-1',id='page1')
        self.model=SubElement(self.diagram,'mxGraphModel',dx='1600',dy='1000',grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(page_w),pageHeight=str(page_h),math='0',shadow='0')
        self.root=SubElement(self.model,'root')
        SubElement(self.root,'mxCell',id='0')
        SubElement(self.root,'mxCell',id='1',parent='0')
        self.i=2
    def vertex(self,x,y,w,h,text='',style='rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#cbd5e1;fontColor=#111827;fontFamily=Arial;fontSize=18;',parent='1'):
        cid=str(self.i); self.i+=1
        c=SubElement(self.root,'mxCell',id=cid,value=html.escape(text),style=style,vertex='1',parent=parent)
        SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
        return cid
    def edge(self,source,target,text='',style='edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#475569;strokeWidth=2;',parent='1'):
        cid=str(self.i); self.i+=1
        c=SubElement(self.root,'mxCell',id=cid,value=html.escape(text),style=style,edge='1',parent=parent,source=source,target=target)
        SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
        return cid
    def xml(self):
        return '<?xml version="1.0" encoding="UTF-8"?>\n'+tostring(self.mxfile,encoding='unicode')

# palette
NAVY='#0f3b66'; BLUE='#2563eb'; BLUEF='#eff6ff'; GREEN='#16a34a'; GREENF='#f0fdf4'; ORANGE='#ea580c'; ORANGEF='#fff7ed'; RED='#dc2626'; REDF='#fef2f2'; PURPLE='#7c3aed'; PURPLEF='#f5f3ff'; GRAY='#64748b'; GRAYF='#f8fafc'; BORDER='#cbd5e1'; TEXT='#111827'

# ---------- Figure 1 ----------
def make_fig1():
    W,H=1800,1200
    s=SVG(W,H)
    # panels
    panels=[(20,20,1030,570,'A','MAGP architecture',BLUE,BLUEF),
            (1070,20,710,570,'B','Transductive information-access contract',GREEN,GREENF),
            (20,610,1030,570,'C','Mechanism interventions',PURPLE,PURPLEF),
            (1070,610,710,570,'D','Evaluation design and scope',ORANGE,ORANGEF)]
    for x,y,w,h,lbl,title,col,fill in panels:
        s.rect(x,y,w,h,'white',col,3,16)
        s.rect(x,y,w,52,fill,col,0,16)
        s.text(x+18,y+36,lbl,30,700,fill=col)
        s.text(x+60,y+36,title,25,700,fill=NAVY if lbl!='D' else '#7c2d12')
    # A
    x0,y0=45,95
    # rsfmri
    s.rect(x0,y0,145,125,GRAYF,BORDER,2,12)
    s.text(x0+72,y0+28,'Multi-site',18,700,'middle')
    s.text(x0+72,y0+51,'rs-fMRI',18,700,'middle')
    for yy in [y0+78,y0+95,y0+112]:
        s.line(x0+30,yy,x0+115,yy,GRAY,2)
        for k in range(4): s.circle(x0+38+k*22,yy,3,BLUE,BLUE,1)
    s.text(x0+72,y0+148,'ABIDE-I',16,700,'middle')
    # multi-atlas
    s.rect(220,y0,155,125,BLUEF,BLUE,2,12)
    s.text(298,y0+28,'Multi-atlas',18,700,'middle')
    s.text(298,y0+52,'parcellation',18,700,'middle')
    for i,(lab,n) in enumerate([('AAL','116'),('CC200','200'),('HO','context')]):
        s.rect(240,y0+66+i*18,115,15,'white',BORDER,1,4)
        s.text(248,y0+78+i*18,f'{lab} ({n})',13,600)
    s.line(190,y0+62,220,y0+62,GRAY,2,arrow=True)
    # FC + fold-local preprocessing
    s.rect(405,y0,170,125,'#f8fafc',GRAY,2,12)
    s.text(490,y0+28,'FC + topology',18,700,'middle')
    # heatmap matrix
    for i in range(6):
        for j in range(6):
            col = '#dbeafe' if (i+j)%3 else '#93c5fd'
            s.rect(438+j*15,y0+48+i*15,13,13,col,'none',0,0)
    s.text(490,y0+150,'Scaler/PCA fit on train',14,600,'middle',fill=GRAY)
    s.line(375,y0+62,405,y0+62,GRAY,2,arrow=True)
    # SubjectGNN
    s.rect(610,y0-10,195,285,GREENF,GREEN,2,12)
    s.text(707,y0+18,'SubjectGNN',19,700,'middle')
    for k in range(3):
        s.rect(646,y0+43+k*38,122,28,'white',GREEN,1,6)
        s.text(707,y0+63+k*38,f'GCN layer {k+1}',14,600,'middle')
        if k<2: s.line(707,y0+71+k*38,707,y0+81+k*38,GREEN,2,arrow=True)
    s.line(707,y0+154,665,y0+179,GREEN,2,arrow=True)
    s.line(707,y0+154,750,y0+179,GREEN,2,arrow=True)
    s.rect(625,y0+181,80,42,'white',GREEN,1,7); s.text(665,y0+207,'SAGPool',13,600,'middle')
    s.rect(713,y0+181,80,42,'white',GREEN,1,7); s.text(753,y0+207,'DiffPool',13,600,'middle')
    s.line(665,y0+223,707,y0+245,GREEN,2,arrow=True); s.line(753,y0+223,707,y0+245,GREEN,2,arrow=True)
    s.rect(652,y0+242,110,30,'white',GREEN,1,6); s.text(707,y0+263,'z_i in R^32',14,700,'middle')
    s.line(575,y0+62,610,y0+62,GRAY,2,arrow=True)
    # population graph
    s.rect(835,y0-10,190,285,ORANGEF,ORANGE,2,12)
    s.text(930,y0+18,'Population graph',19,700,'middle')
    s.multiline(930,y0+44,['site-aware','sex-heterogeneous'],14,600,'middle',fill='#7c2d12',lh=1.15)
    nodes=[(872,y0+115,BLUE),(905,y0+90,BLUE),(942,y0+120,BLUE),(980,y0+93,ORANGE),(999,y0+142,ORANGE),(920,y0+160,GREEN),(875,y0+175,GREEN)]
    # edges
    pairs=[(0,1),(1,2),(2,3),(3,4),(2,5),(5,6),(0,6),(1,5),(4,5)]
    for a,b in pairs:
        xa,ya,_=nodes[a]; xb,yb,_=nodes[b]
        s.line(xa,ya,xb,yb,'#475569',1.6,dash='6 4' if a%2 else None)
    for i,(cx,cy,col) in enumerate(nodes): s.circle(cx,cy,9,col,'white',2)
    s.text(930,y0+205,'site affects topology',14,700,'middle',fill=ORANGE)
    s.text(930,y0+226,'labels do not define edges',13,600,'middle',fill=GRAY)
    s.line(805,y0+252,835,y0+252,GRAY,2,arrow=True)
    # head
    s.rect(45,y0+315,980,145,'#fcfcfd',BORDER,2,12)
    s.text(70,y0+344,'Heterogeneous population head',18,700)
    s.rect(240,y0+365,205,55,BLUEF,BLUE,2,9); s.text(342,y0+389,'Same-sex',14,700,'middle'); s.text(342,y0+410,'TransformerConv',13,600,'middle')
    s.rect(240,y0+426,205,55,REDF,RED,2,9); s.text(342,y0+450,'Different-sex',14,700,'middle'); s.text(342,y0+471,'TransformerConv',13,600,'middle')
    s.rect(525,y0+395,155,60,'white',BORDER,2,9); s.text(603,y0+421,'Adaptive fusion',14,700,'middle'); s.text(603,y0+443,'+ MLP',14,700,'middle')
    s.rect(760,y0+395,170,60,ORANGEF,ORANGE,2,9); s.text(845,y0+421,'ASD probability',14,700,'middle'); s.text(845,y0+444,'p(y=1)',15,700,'middle')
    s.line(445,y0+392,525,y0+413,BLUE,2,arrow=True); s.line(445,y0+454,525,y0+437,RED,2,arrow=True); s.line(680,y0+425,760,y0+425,GRAY,2,arrow=True)
    # B info contract
    bx,by=1095,93
    # mini cohort graph
    s.text(bx,by,'Cohort-visible graph; test labels sealed',18,700)
    train=[(1125,150),(1170,125),(1200,165)]; val=[(1300,145),(1340,180)]; test=[(1450,130),(1510,165),(1565,125)]
    allnodes=train+val+test
    for a in train:
        for b in val[:1]+test[:1]: s.line(*a,*b,'#64748b',1.4)
    for a in val:
        for b in test[:2]: s.line(*a,*b,'#64748b',1.4,dash='5 4')
    for p in train: s.circle(*p,10,BLUE,'white',2)
    for p in val: s.circle(*p,10,GREEN,'white',2)
    for p in test: s.circle(*p,10,ORANGE,'white',2)
    s.text(1158,207,'Train',14,700,'middle',fill=BLUE); s.text(1320,207,'Validation',14,700,'middle',fill=GREEN); s.text(1510,207,'Test',14,700,'middle',fill=ORANGE)
    # lock
    s.rect(1645,122,90,68,REDF,RED,2,10); s.text(1690,147,'Y_test',16,700,'middle',fill=RED); s.text(1690,171,'SEALED',15,700,'middle',fill=RED)
    s.line(1585,154,1645,154,RED,3); s.text(1615,144,'X',24,700,'middle',fill=RED)
    # table
    tx,ty=1095,240; tw=650; rowh=43
    cols=[240,135,135,140]; headers=['Information','Train','Validation','Test']
    x=tx
    for i,cw in enumerate(cols):
        s.rect(x,ty,cw,rowh,'#eaf4ec' if i else '#f8fafc',BORDER,1,0)
        s.text(x+cw/2,ty+28,headers[i],15,700,'middle')
        x+=cw
    rows=[('Imaging features','YES','YES','YES'),('Site / sex metadata','YES','YES','YES'),('ASD / TD labels','YES','selection only','SEALED'),('Scaler / PCA','fit','transform','transform'),('Train-test edges','YES','YES','YES'),('Test-test edges','transductive','transductive','transductive')]
    for r,row in enumerate(rows):
        y=ty+rowh*(r+1); x=tx
        for i,cw in enumerate(cols):
            fill='white' if r%2==0 else '#fafafa'
            if row[0]=='ASD / TD labels' and i==3: fill=REDF
            s.rect(x,y,cw,rowh,fill,BORDER,1,0)
            val=row[i]
            col=RED if val=='SEALED' else TEXT
            s.text(x+cw/2,y+27,val,14,700 if i>0 else 600,'middle',fill=col)
            x+=cw
    s.multiline(1095,557,['Key distinction: transductive context visibility is permitted;', 'test-label access is not.'],15,700,fill=NAVY,lh=1.15)
    # C interventions
    cx0,cy0=45,675
    cards=[]
    titles=[('C1','Constant features'),('C2','Within-site shuffle'),('C3','Remove train-test'),('C4','Degree-preserving'),('C5','Same-site sup. mask')]
    cardw=190; gap=12
    for i,(lab,title) in enumerate(titles):
        x=cx0+i*(cardw+gap); y=cy0
        s.rect(x,y,cardw,445,'white',BORDER,2,11)
        s.rect(x,y,cardw,48,PURPLEF,PURPLE,0,11)
        s.text(x+10,y+31,lab,14,700,fill=PURPLE); s.text(x+44,y+31,title,12,700)
        cards.append((x,y))
    # C1 icon
    x,y=cards[0]
    pts=[(x+45,y+115),(x+95,y+88),(x+145,y+115),(x+75,y+155),(x+130,y+165)]
    for a,b in [(0,1),(1,2),(0,3),(3,4),(2,4),(1,3)]: s.line(*pts[a],*pts[b],GRAY,1.5)
    for p in pts: s.circle(*p,8,'#d1d5db',GRAY,1.5)
    s.text(x+95,y+195,'x_i = c',17,700,'middle',fill=GRAY)
    s.multiline(x+95,y+245,['Keeps graph;', 'removes imaging', 'variation.'],14,600,'middle',fill=GRAY,lh=1.2)
    # C2
    x,y=cards[1]
    pts=[(x+45,y+110),(x+95,y+85),(x+145,y+110),(x+70,y+155),(x+130,y+155)]
    for a,b in [(0,1),(1,2),(0,3),(3,4),(2,4),(1,3)]: s.line(*pts[a],*pts[b],BLUE,1.5)
    cols=[BLUE,GREEN,ORANGE,BLUE,GREEN]
    for p,col in zip(pts,cols): s.circle(*p,8,col,'white',1.5)
    s.text(x+95,y+194,'x1 x2 x3',14,700,'middle')
    s.text(x+95,y+218,'shuffle within site',14,700,'middle',fill=PURPLE)
    s.multiline(x+95,y+265,['Preserves site;', 'breaks subject-', 'imaging pairing.'],14,600,'middle',fill=GRAY,lh=1.2)
    # C3
    x,y=cards[2]
    left=[(x+45,y+105),(x+65,y+145),(x+95,y+115)]; right=[(x+135,y+100),(x+155,y+145)]
    for a,b in [(0,1),(1,2),(0,2)]: s.line(*left[a],*left[b],BLUE,1.5)
    s.line(*right[0],*right[1],ORANGE,1.5)
    for p in left:s.circle(*p,8,BLUE,'white',1.5)
    for p in right:s.circle(*p,8,ORANGE,'white',1.5)
    s.line(x+100,y+125,x+130,y+125,RED,3); s.text(x+115,y+119,'X',22,700,'middle',fill=RED)
    s.multiline(x+95,y+245,['Cuts the', 'supervised-to-test', 'message path.'],14,600,'middle',fill=GRAY,lh=1.2)
    # C4
    x,y=cards[3]
    pts=[(x+40,y+105),(x+85,y+82),(x+140,y+105),(x+65,y+155),(x+130,y+155)]
    for a,b in [(0,1),(1,2),(2,4),(4,3),(3,0),(1,3)]: s.line(*pts[a],*pts[b],GRAY,1.5)
    for p in pts:s.circle(*p,8,'#e5e7eb',GRAY,1.5)
    s.text(x+95,y+193,'edge swap',14,700,'middle',fill=PURPLE)
    s.text(x+95,y+217,'degree preserved',14,700,'middle')
    s.multiline(x+95,y+265,['Keeps degree;', 'destroys specific', 'site topology.'],14,600,'middle',fill=GRAY,lh=1.2)
    # C5
    x,y=cards[4]
    pts=[(x+45,y+100),(x+80,y+145),(x+120,y+100),(x+150,y+145)]
    for a,b in [(0,1),(1,2),(2,3),(0,2)]: s.line(*pts[a],*pts[b],GRAY,1.5)
    for p in pts[:2]: s.circle(*p,8,BLUE,'white',1.5)
    for p in pts[2:]: s.circle(*p,8,ORANGE,'white',1.5)
    s.text(x+95,y+190,'same-site labels',14,700,'middle')
    s.text(x+95,y+215,'masked from loss',14,700,'middle',fill=RED)
    s.multiline(x+95,y+263,['Tests whether', 'same-site', 'supervision matters.'],14,600,'middle',fill=GRAY,lh=1.2)
    # D
    dx,dy=1095,675
    # in-cohort
    s.rect(dx,dy,300,205,BLUEF,BLUE,2,10)
    s.text(dx+150,dy+30,'In-cohort transductive',16,700,'middle')
    s.text(dx+150,dy+53,'evaluation',16,700,'middle')
    sites=[(dx+65,dy+115),(dx+150,dy+105),(dx+230,dy+120)]
    for si,(sx,sy) in enumerate(sites):
        s.circle(sx-12,sy,8,BLUE,'white',1.5); s.circle(sx+12,sy-12,8,ORANGE,'white',1.5); s.circle(sx+16,sy+16,8,GREEN,'white',1.5)
        s.line(sx-12,sy,sx+12,sy-12,GRAY,1); s.line(sx-12,sy,sx+16,sy+16,GRAY,1)
        s.text(sx,dy+175,f'Site {si+1}',12,600,'middle')
    # LOSO
    s.rect(dx+320,dy,355,205,ORANGEF,ORANGE,2,10)
    s.text(dx+497,dy+30,'Leave-one-site-out (LOSO)',16,700,'middle')
    s.text(dx+410,dy+68,'Train on 19 sites',13,700,'middle',fill=BLUE)
    s.text(dx+580,dy+68,'Test on 1 unseen site',13,700,'middle',fill=ORANGE)
    for k in range(6): s.circle(dx+360+(k%3)*32,dy+112+(k//3)*35,7,BLUE,'white',1.5)
    s.line(dx+475,dy+95,dx+475,dy+165,GRAY,2,dash='5 4')
    for k in range(4): s.circle(dx+555+(k%2)*34,dy+115+(k//2)*35,7,ORANGE,'white',1.5)
    # cross architecture
    s.rect(dx,dy+225,675,188,'white',BORDER,2,10)
    s.text(dx+18,dy+253,'Cross-architecture scoping controls',16,700)
    controls=['MAGP dual-channel','Single-channel Transformer','GAT','GCN','MLP','Parisot-GCN']
    for i,c in enumerate(controls):
        xx=dx+18+(i%3)*215; yy=dy+275+(i//3)*54
        fill=BLUEF if i==0 else '#f8fafc'
        stroke=BLUE if i==0 else BORDER
        s.rect(xx,yy,195,38,fill,stroke,1.5,7); s.text(xx+97,yy+25,c,12,700 if i==0 else 600,'middle')
    s.text(dx+18,dy+400,'Primary outcome: ROC-AUC with paired, site-aware uncertainty analysis.',14,700,fill=NAVY)
    svg=SRC/'Figure1_study_design.svg'; svg.write_text(s.finish(),encoding='utf-8')

    # drawio approximate editable source
    d=Drawio(W,H)
    # panel containers and headings
    for x,y,w,h,lbl,title,col,fill in panels:
        style=f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={col};strokeWidth=2;fontFamily=Arial;fontSize=18;'
        d.vertex(x,y,w,h,'',style)
        hstyle=f'rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={col};strokeWidth=1;fontColor={NAVY};fontFamily=Arial;fontSize=22;fontStyle=1;align=left;spacingLeft=12;'
        d.vertex(x,y,w,52,f'{lbl}   {title}',hstyle)
    # Add editable major boxes, not every decorative node, but all content remains editable.
    # A flow
    ids=[]
    for (x,y,w,h,t,fc,sc) in [
        (45,95,145,125,'Multi-site rs-fMRI\nABIDE-I',GRAYF,BORDER),
        (220,95,155,125,'Multi-atlas parcellation\nAAL / CC200 / HO',BLUEF,BLUE),
        (405,95,170,125,'FC + topology\nScaler/PCA fit on train','#f8fafc',GRAY),
        (610,85,195,285,'SubjectGNN\nGCN x3\nSAGPool + DiffPool\nz_i in R^32',GREENF,GREEN),
        (835,85,190,285,'Population graph\nsite-aware\nsex-heterogeneous\nsite affects topology',ORANGEF,ORANGE),
    ]:
        st=f'rounded=1;whiteSpace=wrap;html=1;fillColor={fc};strokeColor={sc};strokeWidth=2;fontFamily=Arial;fontSize=16;'
        ids.append(d.vertex(x,y,w,h,t,st))
    for a,b in zip(ids,ids[1:]): d.edge(a,b)
    # head branches
    headbox=d.vertex(45,410,980,145,'Heterogeneous population head',f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={BORDER};fontFamily=Arial;fontSize=17;fontStyle=1;align=left;spacingLeft=12;')
    ss=d.vertex(240,460,205,45,'Same-sex TransformerConv',f'rounded=1;whiteSpace=wrap;html=1;fillColor={BLUEF};strokeColor={BLUE};fontFamily=Arial;fontSize=14;')
    ds=d.vertex(240,512,205,45,'Different-sex TransformerConv',f'rounded=1;whiteSpace=wrap;html=1;fillColor={REDF};strokeColor={RED};fontFamily=Arial;fontSize=14;')
    fu=d.vertex(525,480,155,55,'Adaptive fusion + MLP',f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={BORDER};fontFamily=Arial;fontSize=14;')
    pr=d.vertex(760,480,170,55,'ASD probability\np(y=1)',f'rounded=1;whiteSpace=wrap;html=1;fillColor={ORANGEF};strokeColor={ORANGE};fontFamily=Arial;fontSize=14;fontStyle=1;')
    d.edge(ss,fu); d.edge(ds,fu); d.edge(fu,pr)
    # B matrix as editable text blocks
    d.vertex(1095,93,650,120,'Cohort-visible graph\nTrain (labeled)  -  Validation (selection only)  -  Test (unlabeled)\nY_test SEALED',f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={BORDER};fontFamily=Arial;fontSize=16;')
    d.vertex(1095,240,650,300,'Information access\n\nImaging features: YES / YES / YES\nSite / sex metadata: YES / YES / YES\nASD / TD labels: YES / selection only / SEALED\nScaler / PCA: fit / transform / transform\nTrain-test edges: YES / YES / YES\nTest-test edges: transductive / transductive / transductive',f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={GREEN};fontFamily=Arial;fontSize=15;align=left;spacingLeft=12;')
    # C intervention boxes
    for i,(lab,title) in enumerate(titles):
        x=45+i*202
        d.vertex(x,675,190,445,f'{lab}  {title}\n\nMechanism intervention',f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={PURPLE};fontFamily=Arial;fontSize=15;fontStyle=1;')
    # D boxes
    d.vertex(1095,675,300,205,'In-cohort transductive evaluation\nSites contain train + validation + test subjects',f'rounded=1;whiteSpace=wrap;html=1;fillColor={BLUEF};strokeColor={BLUE};fontFamily=Arial;fontSize=15;')
    d.vertex(1415,675,355,205,'Leave-one-site-out (LOSO)\nTrain on 19 sites\nTest on 1 unseen site',f'rounded=1;whiteSpace=wrap;html=1;fillColor={ORANGEF};strokeColor={ORANGE};fontFamily=Arial;fontSize=15;')
    d.vertex(1095,900,675,188,'Cross-architecture scoping controls\nMAGP dual-channel | Single Transformer | GAT | GCN | MLP | Parisot-GCN\n\nPrimary outcome: ROC-AUC with paired, site-aware uncertainty analysis.',f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={BORDER};fontFamily=Arial;fontSize=15;align=left;spacingLeft=12;')
    (SRC/'Figure1_study_design.drawio').write_text(d.xml(),encoding='utf-8')

    return svg

# ---------- Figure S1: protocol schematic ----------
def make_figS1():
    W,H=1500,620
    s=SVG(W,H)
    s.text(28,42,'Protocol schematic: in-cohort transduction, edge-path ablation, and LOSO',28,700,fill=NAVY)
    panel_specs=[(25,75,460,510,'(a) In-cohort transductive graph',BLUE,BLUEF),(520,75,460,510,'(b) Remove train-test edges',PURPLE,PURPLEF),(1015,75,460,510,'(c) Leave-one-site-out (LOSO)',ORANGE,ORANGEF)]
    for x,y,w,h,title,col,fill in panel_specs:
        s.rect(x,y,w,h,'white',col,3,15); s.rect(x,y,w,55,fill,col,0,15); s.text(x+18,y+36,title,21,700,fill=NAVY)
    # a graph
    x,y=25,75
    train=[(95,230),(165,175),(220,255),(145,320)]; test=[(325,190),(375,285),(305,345)]
    for a,b in [(0,1),(1,2),(2,3),(3,0),(1,3)]: s.line(*train[a],*train[b],BLUE,2)
    for a in train:
        for b in test[:2]: s.line(*a,*b,'#64748b',1.6,dash='6 4')
    for a,b in [(0,1),(1,2)]: s.line(*test[a],*test[b],ORANGE,1.6,dash='6 4')
    for p in train:s.circle(*p,12,BLUE,'white',2)
    for p in test:s.circle(*p,12,ORANGE,'white',2)
    s.text(155,395,'Training/reference nodes',16,700,'middle',fill=BLUE); s.text(340,395,'Test nodes',16,700,'middle',fill=ORANGE)
    s.multiline(255,455,['Train-test and test-test edges are present.', 'Test labels remain sealed.'],16,600,'middle',fill=GRAY,lh=1.25)
    # b
    x,y=520,75
    train=[(590,230),(660,175),(715,255),(640,320)]; test=[(820,190),(870,285),(800,345)]
    for a,b in [(0,1),(1,2),(2,3),(3,0),(1,3)]: s.line(*train[a],*train[b],BLUE,2)
    for a,b in [(0,1),(1,2)]: s.line(*test[a],*test[b],ORANGE,1.6,dash='6 4')
    for p in train:s.circle(*p,12,BLUE,'white',2)
    for p in test:s.circle(*p,12,ORANGE,'white',2)
    s.line(736,245,775,245,RED,4); s.text(755,236,'X',28,700,'middle',fill=RED)
    s.text(655,395,'Training/reference nodes',16,700,'middle',fill=BLUE); s.text(835,395,'Test nodes',16,700,'middle',fill=ORANGE)
    s.multiline(750,455,['All train-test edges are removed.', 'Tests whether supervised-to-test message paths are necessary.'],15,600,'middle',fill=GRAY,lh=1.25)
    # c
    x,y=1015,75
    # train 19 sites cluster and unseen site
    for k in range(8):
        cx=1085+(k%4)*45; cy=205+(k//4)*55
        s.circle(cx,cy,10,BLUE,'white',2)
        if k>0: s.line(1085+((k-1)%4)*45,205+((k-1)//4)*55,cx,cy,'#94a3b8',1.2)
    s.line(1255,165,1255,380,GRAY,2,dash='7 5')
    for k in range(4):
        cx=1340+(k%2)*55; cy=220+(k//2)*70
        s.circle(cx,cy,12,ORANGE,'white',2)
        if k>0: s.line(1340+((k-1)%2)*55,220+((k-1)//2)*70,cx,cy,ORANGE,1.5,dash='6 4')
    s.text(1160,410,'Train: 19 sites',16,700,'middle',fill=BLUE); s.text(1370,410,'Test: one unseen site',16,700,'middle',fill=ORANGE)
    s.multiline(1245,465,['No training subject shares the target site.', 'Measures unseen-site generalization, not query isolation.'],15,600,'middle',fill=GRAY,lh=1.25)
    # legend
    s.circle(55,600,7,BLUE,'white',1.5); s.text(70,606,'training/reference',14,600)
    s.circle(220,600,7,ORANGE,'white',1.5); s.text(235,606,'test',14,600)
    s.line(330,600,380,600,GRAY,1.5,dash='6 4'); s.text(390,606,'transductive edge',14,600)
    svg=SRC/'FigureS1_protocol_schematic.svg'; svg.write_text(s.finish(),encoding='utf-8')
    # drawio editable source
    d=Drawio(W,H)
    for x,y,w,h,title,col,fill in panel_specs:
        d.vertex(x,y,w,h,title,f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={col};strokeWidth=2;fontFamily=Arial;fontSize=20;fontStyle=1;verticalAlign=top;spacingTop=12;')
    d.vertex(55,160,390,320,'Training/reference nodes + test nodes\n\nTrain-test and test-test edges present\nTest labels sealed',f'rounded=1;whiteSpace=wrap;html=1;fillColor={BLUEF};strokeColor={BLUE};fontFamily=Arial;fontSize=16;')
    d.vertex(550,160,390,320,'Training/reference nodes + test nodes\n\nTrain-test edges removed\nTests the message pathway',f'rounded=1;whiteSpace=wrap;html=1;fillColor={PURPLEF};strokeColor={PURPLE};fontFamily=Arial;fontSize=16;')
    d.vertex(1045,160,390,320,'Train: 19 sites\nTest: 1 unseen site\n\nNo same-site training subjects\nUnseen-site generalization',f'rounded=1;whiteSpace=wrap;html=1;fillColor={ORANGEF};strokeColor={ORANGE};fontFamily=Arial;fontSize=16;')
    (SRC/'FigureS1_protocol_schematic.drawio').write_text(d.xml(),encoding='utf-8')
    return svg

fig1=make_fig1(); figs1=make_figS1()

# Export SVG -> PDF and PNG
for svg, pdf, png in [
    (SRC/'Figure1_study_design.svg', MAIN/'Figure1_study_design.pdf', PRE/'Figure1_study_design.png'),
    (SRC/'FigureS1_protocol_schematic.svg', SUP/'FigureS1_protocol_schematic.pdf', PRE/'FigureS1_protocol_schematic.png')
]:
    subprocess.run(['inkscape',str(svg),'--export-type=pdf','--export-filename='+str(pdf)],check=True)
    subprocess.run(['inkscape',str(svg),'--export-type=png','--export-width=2400','--export-filename='+str(png)],check=True)

# Copy SVG derivatives into final folders too
shutil.copy2(SRC/'Figure1_study_design.svg', MAIN/'Figure1_study_design.svg')
shutil.copy2(SRC/'FigureS1_protocol_schematic.svg', SUP/'FigureS1_protocol_schematic.svg')

# Supplementary attribution: preserve original data/artwork unchanged, standardized file name.
orig=Path('/mnt/data/Global_Average(1).pdf')
if orig.exists():
    shutil.copy2(orig, SUP/'FigureS3_attribution_map.pdf')
    # render preview
    subprocess.run(['python','/home/oai/skills/pdfs/scripts/render_pdf.py',str(SUP/'FigureS3_attribution_map.pdf'),'--out_dir',str(OUT/'_attrib_render'),'--dpi','220'],check=True)
    candidates=list((OUT/'_attrib_render').glob('*.png'))
    if candidates:
        shutil.copy2(candidates[0], PRE/'FigureS3_attribution_map.png')

# Captions and usage notes
captions='''# Paper M figure captions and placement\n\n## Figure 1 — main manuscript\n**Overview of the MAGP architecture, transductive information-access setting, mechanism interventions, and evaluation design.**\n(A) Multi-site rs-fMRI is transformed into multi-atlas functional-connectivity representations, processed by fold-local scaling/PCA, and encoded by a SubjectGNN with three GCN layers plus SAGPool and DiffPool to obtain a 32-dimensional subject embedding. A site-aware, sex-heterogeneous population graph is then processed by separate same-sex and different-sex TransformerConv channels followed by adaptive fusion and node-level classification. (B) Information-access contract for the transductive setting: imaging features and site/sex metadata are visible for all nodes, whereas test labels remain sealed; scaler/PCA parameters are fit on training data and only transformed elsewhere. (C) Targeted mechanism interventions used to localize the source of high cohort-level performance: constant features, within-site feature shuffling, removal of train-test edges, degree-preserving rewiring, and same-site supervision masking. (D) Evaluation scope: in-cohort transductive cross-validation, leave-one-site-out generalization, and architecture-matched scoping controls.\n\nRecommended placement: end of Methods overview / start of experimental design.\n\n## Figure S1 — supplementary methods\n**Protocol schematic for in-cohort transduction, train-test edge-path ablation, and leave-one-site-out evaluation.**\n(A) In-cohort transductive evaluation permits train-test and test-test graph edges while test labels remain sealed. (B) The train-test edge intervention removes all edges linking training/reference nodes to test nodes while preserving within-partition graph structure. (C) Leave-one-site-out evaluation trains on subjects from all non-target sites and evaluates on one entirely unseen site, thereby testing unseen-site generalization rather than query isolation.\n\nRecommended placement: Supplementary Methods, referenced from the protocol section.\n\n## Figure S3 — supplementary interpretability\n**Global-average model-attribution maps.**\nUse only after the attribution back-projection and cross-fold/cross-seed stability analyses are finalized. The file in this package is the previously supplied artwork renamed for supplementary use; its underlying data have not been altered. Avoid biomarker or causal wording.\n'''
(OUT/'FIGURE_CAPTIONS.md').write_text(captions,encoding='utf-8')

readme='''# Paper M figure package\n\nThis package follows the current Paper M mechanism/deconfounding storyline.\n\n## Recommended source-of-truth workflow\n- Edit `source/*.drawio` in diagrams.net / Draw.io.\n- Use the exported PDF files in LaTeX (`\\includegraphics{...pdf}`).\n- Keep SVG as a vector derivative/backup.\n- PNG files are previews only.\n\n## Main manuscript\n- `main/Figure1_study_design.pdf` — recommended replacement for the legacy model/protocol figures.\n\n## Supplementary material\n- `supplementary/FigureS1_protocol_schematic.pdf` — reframes the old full/semi/inductive schematic into the current mechanism-oriented protocol comparison.\n- `supplementary/FigureS3_attribution_map.pdf` — the supplied Global_Average artwork preserved unchanged and repositioned as supplementary interpretability material.\n\n## Editing notes\n1. Do not reintroduce the old Full/Semi/Inductive deployment ladder into Paper M; that story belongs to the evaluation-regime audit.\n2. In Figure 1, site is shown as part of population-graph topology rather than as a diagnostic label.\n3. Test labels are explicitly sealed; transductive context visibility is not equivalent to test-label leakage.\n4. Figure S3 should remain supplementary until attribution stability across folds/seeds is documented.\n\nAll scientific labels are based on the user-supplied manuscript/figures and the current Paper M study design; no numerical results were invented in these schematics.\n'''
(OUT/'README.md').write_text(readme,encoding='utf-8')

print('generated', OUT)
