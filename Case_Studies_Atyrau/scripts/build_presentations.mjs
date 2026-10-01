// Rebuild research decks as editable academic slides from evidence-backed JSON.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const api=process.env.ARTIFACT_TOOL_ENTRY?await import(pathToFileURL(process.env.ARTIFACT_TOOL_ENTRY).href):await import('@oai/artifact-tool');
if(!process.env.RUNTIME_NODE_MODULES&&process.env.ARTIFACT_TOOL_ENTRY)process.env.RUNTIME_NODE_MODULES=path.resolve(path.dirname(process.env.ARTIFACT_TOOL_ENTRY),'../../..');
const {Presentation,PresentationFile}=api;
const selected=process.argv[2]||'all';
const cases=(await fs.readdir(root)).filter(x=>/^0[1-4]_/.test(x)&&(selected==='all'||selected.split(',').some(n=>x.startsWith(n)))).sort();
const theme={ink:'#173342',muted:'#516471',paper:'#FFFFFF',navy:'#102B3A',light:'#F1F5F7'};
const accents={'01':'#087E8B','02':'#2864AD','03':'#28775A','04':'#AE562D'};
const caseNumber={'01':'1','02':'10','03':'12','04':'15'};
const family='Arial';
function text(slide,value,x,y,w,h,size=28,bold=false,color=theme.ink){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=Array.isArray(value)?value.join('\n\n'):String(value||'');
 s.text.style={typeface:family,fontSize:size,bold,color,autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0}};
 return s;
}
function shortBody(s){return Array.isArray(s.body)?s.body.join('\n\n'):String(s.body||'');}
function foot(slide,s,i,count,dark=false){
 const ink=dark?'#A4C1CB':theme.muted;
 if(s.source)text(slide,String(s.source),64,649,1050,44,15,false,ink);
 text(slide,`${i+1} / ${count}`,1156,659,70,25,17,false,ink);
}
function metricBlocks(slide,metrics,accent,y=210){
 const items=metrics,width=1128/items.length;
 if(items.length>4)throw Error('A stat slide supports at most four complete metrics');
 items.forEach((m,i)=>{const x=68+i*width;
  const valueSize=Math.min(items.length===4?57:72,Math.floor((width-32)/(Math.max(1,String(m.value).length)*0.68)));
  text(slide,String(m.value),x,y,width-28,106,valueSize,true,accent);
  text(slide,m.label,x,y+114,width-28,118,items.length===4?23:25,false,theme.ink);
 });
}
function columns(slide,cols,accent,y=197){
 const count=cols.length,width=1128/count;
 cols.forEach((c,i)=>{const x=68+i*width;
  text(slide,c.title,x,y,width-38,65,29,true,accent);
  text(slide,c.body,x,y+82,width-38,315,count>2?25:28,false,theme.ink);
 });
}
for(const c of cases){
 const dir=path.join(root,c);let source=path.join(dir,'slides_v2.json');
 try{await fs.access(source);}catch{source=path.join(dir,'slides.json');}
 let data=JSON.parse(await fs.readFile(source,'utf8'));const slides=data.slides||data;
 if(slides.length<10||slides.length>15)throw Error(`${c}: invalid slide count ${slides.length}`);
 const presentation=Presentation.create({slideSize:{width:1280,height:720}});
 const accent=accents[c.slice(0,2)],nativeTables=[],nativeCharts=[];
 for(let i=0;i<slides.length;i++){
  const s=slides[i],slide=presentation.slides.add(),layout=s.layout||(i===0?'cover':s.figure?'figure':s.chart?'chart':'columns');
  const dark=layout==='cover'||layout==='conclusion';slide.background.fill=dark?theme.navy:theme.paper;
  if(layout==='cover'){
   text(slide,`Исследовательский кейс ${caseNumber[c.slice(0,2)]}`,68,66,1110,45,24,false,'#80CCD2');
   text(slide,s.title,64,155,1132,236,58,true,'#FFFFFF');
   const sub=s.subtitle||shortBody(s);text(slide,sub,68,421,1110,100,29,false,'#DEE9EE');
   if(s.subtitle&&s.body)text(slide,Array.isArray(s.body)?s.body.join(' · '):s.body,68,535,1110,62,20,false,'#C0D5DE');
   text(slide,'Искусственный интеллект и машинное обучение',68,606,1040,32,19,false,'#A4C1CB');
  }else if(layout==='conclusion'){
   text(slide,s.title,64,62,1140,120,43,true,'#FFFFFF');
   text(slide,shortBody(s),68,218,1110,345,34,false,'#EDF4F7');
  }else{
   text(slide,s.title,64,49,1150,118,String(s.title).length>62?39:44,true,theme.ink);
   if(layout==='stat'){
    metricBlocks(slide,s.metrics||[],accent,205);
    if(s.body)text(slide,Array.isArray(s.body)?s.body.join('\n'):shortBody(s),68,462,1100,155,25,false,theme.muted);
   }else if(layout==='figure'){
    if(!s.figure)throw Error(`${c}:${i+1} missing figure`);
    const fig=path.resolve(dir,s.figure);
    const img=await fs.readFile(fig);
    slide.images.add({blob:new Uint8Array(img),contentType:/\.jpe?g$/i.test(fig)?'image/jpeg':'image/png',alt:s.caption||s.title,fit:'contain',position:{left:53,top:185,width:818,height:431}});
    text(slide,shortBody(s),913,197,309,s.link?322:405,25,false,theme.ink);
   }else if(layout==='chart'){
    const ch=s.chart;if(!ch)throw Error(`${c}:${i+1} missing chart`);
    const series=ch.series.map((q,j)=>({...q,values:q.values.map(v=>Number(Number(v).toFixed(3))),fill:j===0?accent:'#2864AD'}));
    const chart=slide.charts.add('bar',{position:{left:58,top:185,width:792,height:416},categories:ch.categories,series,barOptions:{direction:'column',grouping:'clustered',gapWidth:75},hasLegend:series.length>1,legend:{position:'bottom',textStyle:{typeface:family,fontSize:21,fill:theme.muted}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',xAxis:{textStyle:{typeface:family,fontSize:21,fill:theme.muted}},yAxis:{visible:true,min:0,numberFormatCode:'0.000',textStyle:{typeface:family,fontSize:21,fill:theme.muted},title:{text:ch.unit||series.map(q=>q.name).join(' / '),textStyle:{typeface:family,fontSize:21,fill:theme.muted}},majorGridlines:{style:'solid',fill:'#E3E9ED',width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:family,fontSize:22,fill:theme.ink,bold:true}}});
    if(process.env.PRESENTATION_SKILL_DIR){const {applyPresentationChartFont}=await import(pathToFileURL(path.join(process.env.PRESENTATION_SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);applyPresentationChartFont(chart,{fontFamily:family});}
    text(slide,shortBody(s),913,197,309,405,25,false,theme.ink);nativeCharts.push(i+1);
   }else if(layout==='table'){
    const t=s.table,values=[t.headers,...t.rows].map(r=>r.map(String));
    const count=values.length,width=1140,heights=Math.min(370,count*57);
    const widths=t.widths||values[0].map((_,j)=>j===0?width*.36:width*.64/(values[0].length-1));
    const table=slide.tables.add({rows:count,columns:values[0].length,left:68,top:191,width,height:heights,columnWidths:widths,values});
    table.styleOptions={headerRow:true,bandedRows:false};
    table.borders.assign({style:'solid',fill:'#FFFFFF',width:2});
    for(let r=0;r<count;r++)for(let col=0;col<values[0].length;col++){
      const cell=table.getCell(r,col);cell.fill=r===0?theme.navy:(r%2?'#F2F6F8':'#FFFFFF');
      cell.text.style={typeface:family,fontSize:r===0?22:24,bold:r===0,color:r===0?'#FFFFFF':theme.ink};
    }
    if(s.body)text(slide,shortBody(s),68,207+heights,1120,610-(207+heights),25,false,theme.muted);
    nativeTables.push(i+1);
   }else if(layout==='references'){
    if(s.columns)columns(slide,s.columns,accent,185);
    else{
     const refs=Array.isArray(s.body)?s.body:[shortBody(s)];
     const half=Math.ceil(refs.length/2);columns(slide,[{title:'Источники',body:refs.slice(0,half).join('\n\n')},{title:'Методы и данные',body:refs.slice(half).join('\n\n')}],accent,185);
    }
   }else{
    if(s.columns)columns(slide,s.columns,accent,197);
    else if(s.metrics){metricBlocks(slide,s.metrics,accent);if(s.body)text(slide,shortBody(s),68,502,1100,110,27);}
    else text(slide,shortBody(s),68,204,1110,370,31,false,theme.ink);
   }
  }
  if(s.link){
   const box=text(slide,s.link.text||'Открыть видео',layout==='figure'?913:68,574,310,44,25,true,accent);
   box.text=[[{run:s.link.text||'Открыть видео',textStyle:{underline:'sng'},link:{uri:s.link.target,isExternal:true}}]];
  }
  foot(slide,s,i,slides.length,dark);
  slide.speakerNotes.textFrame.setText(String(s.notes||'')+'\n\nИсточники: '+String(s.source||'')+(s.link?'\nЛокальное видео: '+s.link.target:''));
 }
 const build=path.resolve(root,'..','_build','revision_20260926',c);await fs.mkdir(build,{recursive:true});
 const candidatePath=path.join(build,'candidate-'+Date.now()+'.pptx');
 await(await PresentationFile.exportPptx(presentation)).save(candidatePath);
 const finalPath=path.join(dir,'Presentation.pptx');
 if(process.env.PRESENTATION_SKILL_DIR&&process.env.RUNTIME_PYTHON){
  const sk=process.env.PRESENTATION_SKILL_DIR,{finalizePresentation}=await import(pathToFileURL(path.join(sk,'container_tools/artifact_tool_utils.mjs')).href);
  const publishedDir=path.join(build,'validated');await fs.mkdir(publishedDir,{recursive:true});
  const checked=path.join(publishedDir,'checked-'+Date.now()+'.pptx');
  const tableLayoutPolicy=nativeTables.flatMap(n=>['--require-native-table-slide',String(n)]);
  await finalizePresentation({workspaceDir:path.resolve(root,'..'),candidatePath,finalPath:checked,pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(sk,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(sk,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tableLayoutPolicy],explicitTotalSlideCount:slides.length,requiredNativeTableOwnerSlides:nativeTables,requiredNativeChartOwnerSlides:nativeCharts,fontPolicy:{basis:'design',families:[family]},materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,receiptPath:path.join(build,'validation-'+Date.now()+'.json')});
  await fs.copyFile(checked,finalPath);
 }else await fs.copyFile(candidatePath,finalPath);
 for(let i=0;i<presentation.slides.items.length;i++){
  const png=await presentation.export({slide:presentation.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
 }
 await fs.writeFile(path.join(dir,'results','presentation_structure.json'),JSON.stringify({revision:'2026-09-26',source:path.basename(source),slides:slides.length,native_chart_slides:nativeCharts,native_table_slides:nativeTables,font:family,slide_size:[1280,720]},null,2));
 console.log(c,slides.length,'slides',nativeCharts.length,'native charts',nativeTables.length,'native tables');
}
