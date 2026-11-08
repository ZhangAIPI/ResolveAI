"""Read-only analysis of already-recorded ResolveAI episodes. No inference."""
from pathlib import Path
from contextlib import closing
from collections import Counter
import sqlite3, json, hashlib
import numpy as np
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('repository',type=Path)
parser.add_argument('runtime',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--include-records',action='store_true',help='Include case-level trace provenance; default is compact totals plus aggregate digests.')
args=parser.parse_args()
R=args.repository
RT=args.runtime
P=RT/'evaluation/cvpr-paper-priority'
O=RT/'evaluation/cvpr-20261101'
Q=RT/'evaluation/cvpr-20261102-controls/text384'
OUT=args.output.parent
OUT.mkdir(parents=True,exist_ok=True)
REF=json.loads((R/'docs/results/cvpr_20261102_candidate_scores.json').read_text())
F=set(REF['family_ids'])
assert len(F)==209

DIRS={
 '2B Base':P/'qwen3_2b',
 '2B SFT':P/'qwen3_2b_mixed_sft/benchmark',
 '2B SFT+RL':P/'qwen3_2b_grpo_final/benchmark',
 '7B SFT+RL':P/'qwen25_grpo_final/benchmark',
 '8B Base':O/'qwen3',
 '8B SFT':P/'qwen3_mixed_sft/benchmark',
 '8B RL':P/'qwen3_8b_rl_only_final/benchmark',
 '8B SFT+RL':P/'qwen3_8b_grpo_final/benchmark',
}
def render_main_search_behavior(result):
    """Render the representative policies from the recorded behavior totals."""
    rows=(
        ("2B SFT+RL","Qwen3-VL-2B","SFT+RL"),
        ("7B SFT+RL","Qwen2.5-VL-7B","SFT+RL"),
        ("8B Base","Qwen3-VL-8B","Base"),
        ("8B RL","Qwen3-VL-8B","RL"),
        ("8B SFT+RL","Qwen3-VL-8B","SFT+RL"),
    )
    lines=[
      '\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{5pt}',
      '\\begin{minipage}{.84\\textwidth}\\centering',
      '\\caption{\\textbf{Photograph requests and stopping.} Obtainable distinguishes requesting a photograph, receiving new pixels and submitting a valid answer. Unavailable separates definite answers from explicit Need more evidence responses. All rates include every attempted episode; the remainder in Unavailable represents failed submissions.}',
      '\\label{tab:search-behavior}',
      '\\begin{tabular}{@{}llrrrrr@{}}\\toprule',
      ' & & \\multicolumn{3}{c}{Obtainable (\\%)} & \\multicolumn{2}{c}{Unavailable (\\%)}\\\\',
      '\\cmidrule(lr){3-5}\\cmidrule(l){6-7}',
      'Model & Training & Requests & Photo retrieval & Finished & Definite & Need more\\\\\\midrule',
    ]
    for name,model,training in rows:
        obtainable=result["policies"][name]["Obtainable"]["overall"]["totals"]
        unavailable=result["policies"][name]["Unavailable"]["overall"]["totals"]
        values=[100*obtainable[key]/obtainable["episodes"]
                for key in ("requested","novel_image_request","finished")]
        values += [100*unavailable[key]/unavailable["episodes"]
                   for key in ("definite","abstained")]
        lines.append(model+" & "+training+" & "+" & ".join(f"{v:.1f}" for v in values)+r"\\")
    return "\n".join(lines+[r"\bottomrule\end{tabular}\end{minipage}\end{table*}"])+"\n"


def load(d,only_interactive=True):
    files=[d/'episodes.sqlite'] if (d/'episodes.sqlite').exists() else sorted(d.glob('shard-*/episodes.sqlite'))
    assert files,d
    out={}
    sql='SELECT result,trace FROM episodes'
    if only_interactive:sql+=" WHERE arm='interactive_agent'"
    for f in files:
        with closing(sqlite3.connect(f.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            for raw,tr in db.execute(sql):
                row=json.loads(raw)
                row.setdefault('source_label_match',row.get('source_label') in ['Supported','Refuted'] and row.get('verdict')==row.get('source_label'))
                if row['family_id'] not in F:continue
                key=(row['family_id'],row['variant'],row['arm'])
                assert key not in out,(d,key)
                out[key]=(row,json.loads(tr),str(f.relative_to(RT)),hashlib.sha256(tr.encode()).hexdigest())
    return out

def content_parts(m):
    value=m.get('content',[])
    return value if isinstance(value,list) else []

def request_info(tr):
    seen=set()
    records=[]
    names={}
    for m in tr['messages']:
        if m.get('role')=='assistant':
            for c in m.get('tool_calls',[]):
                fun=c.get('function',{})
                names[c.get('id')]=fun
        imgs={Path(x['image_ref']).name for x in content_parts(m) if x.get('type')=='image' and x.get('image_ref')}
        if m.get('role')=='tool' and (m.get('name')=='request_photo' or names.get(m.get('tool_call_id'),{}).get('name')=='request_photo'):
            objs=[]
            for part in content_parts(m):
                if part.get('type')=='text':
                    try:objs.append(json.loads(part['text']))
                    except (ValueError,TypeError):pass
            status=next((x.get('status') for x in objs if isinstance(x,dict) and 'status' in x),None)
            arg=names.get(m.get('tool_call_id'),{}).get('arguments',{})
            if isinstance(arg,str):
                try:arg=json.loads(arg)
                except ValueError:arg={}
            records.append({'status':status,'novel_images':len(imgs-seen),'returned_images':len(imgs),
               'repeated_only':bool(imgs) and not bool(imgs-seen),'query':arg.get('query',{}),
               'image_refs':sorted(imgs)})
        seen.update(imgs)
    return records

def behavior(data,variant,task=None):
    records=[]
    for (f,v,a),(row,tr,path,sha) in data.items():
        if v!=variant or (task and row['task']!=task):continue
        req=request_info(tr)
        failure=any(x['status'] in ['not_found','unable_to_provide','unavailable','missing'] for x in req)
        records.append({
          'family_id':f,'task':row['task'],'request_calls':row['requests'],
          'recorded_request_responses':len(req),'requested':row['requests']>0,
          'novel_image_request':any(x['novel_images']>0 for x in req),
          'novel_images':sum(x['novel_images'] for x in req),
          'repeated_image_request':any(x['repeated_only'] for x in req),
          'repeated_image_calls':sum(x['repeated_only'] for x in req),
          'failed_request':failure,
          'failed_then_definite':failure and row['verdict'] in ['Supported','Refuted'],
          'abstained':row['verdict']=='Need more evidence',
          'finished':row['termination']=='finished',
          'definite':row['verdict'] in ['Supported','Refuted'],
          'source_label_match':row['source_label_match'],
          'request_records':req,'trace_sha256':sha,'source_database':path,
        })
    n=len(records)
    expected=209 if task is None else {'state':153,'identity':56}[task]
    assert n==expected,(variant,task,n)
    keys=['request_calls','recorded_request_responses','requested','novel_image_request','novel_images',
       'repeated_image_request','repeated_image_calls','failed_request','failed_then_definite',
       'abstained','finished','definite','source_label_match']
    totals={k:sum(int(x[k]) for x in records) for k in keys}
    totals['episodes']=n
    totals['request_views']=dict(Counter(q['query'].get('view','<unspecified>') for x in records for q in x['request_records']))
    totals['request_status']=dict(Counter(str(q['status']) for x in records for q in x['request_records']))
    totals['failure_then_definite_fraction']=(totals['failed_then_definite']/totals['failed_request'] if totals['failed_request'] else None)
    return {'totals':totals,'records':records}

def paired(a,b,task=None):
    out=Counter();changes=[]
    for f in sorted(F):
        x=a[(f,'Sufficient','interactive_agent')][0]
        y=b[(f,'Sufficient','interactive_agent')][0]
        if task and x['task']!=task:continue
        xx,yy=bool(x['source_label_match']),bool(y['source_label_match'])
        out[('correct' if xx else 'wrong')+'_'+('correct' if yy else 'wrong')]+=1
        changes.append(int(yy)-int(xx))
    changes=np.asarray(changes)
    rng=np.random.default_rng(20261105)
    boot=100*changes[rng.integers(len(changes),size=(10000,len(changes)))].mean(axis=1)
    return {'families':len(changes),'counts':dict(out),'difference_pp':float(changes.mean()*100),
       'bootstrap_95_ci_pp':np.percentile(boot,[2.5,97.5]).tolist(),'seed':20261105,'resamples':10000,
       'direction':'SFT+RL minus direct RL'}

all_data={k:load(v) for k,v in DIRS.items()}
result={'scope':'Observed actions and source-label agreement only. Photo retrieval is new image_ref pixel hash relative to all prior observations, not reviewed sufficient evidence.',
'dataset_sha256':REF['dataset_sha256'],'family_ids':sorted(F),
'policies':{},'paired_8b_rl_to_sftrl':{},'text384_controls':{}}
for name,data in all_data.items():
    assert len(data)==209*4,(name,len(data))
    result['policies'][name]={v:{t:behavior(data,v,None if t=='overall' else t) for t in ['overall','state','identity']}
       for v in ['Sufficient','Obtainable','Missing','Unavailable']}
for task in ['overall','state','identity']:
    result['paired_8b_rl_to_sftrl'][task]=paired(all_data['8B RL'],all_data['8B SFT+RL'],None if task=='overall' else task)

def prompt(tr):
    messages=[]
    for m in tr.get('messages',[]):
        if m.get('role')=='assistant':break
        messages.append(m)
    return {'messages':messages,'tools':tr.get('tools')}

for name in ['qwen25','qwen3','internvl35']:
    a=load(O/name,False); b=load(Q/name,False)
    ma=json.loads((O/name/'manifest.json').read_text())
    mb=json.loads((Q/name/'manifest.json').read_text())
    ca,cb=ma['configuration'],mb['configuration']
    samefields=['plan_sha256','dataset_sha256','source_sha256','model_id','revision','max_context_tokens','image_max_pixels','tool_adapter','assistant_prefix']
    checks={k:ca.get(k)==cb.get(k) for k in samefields}
    checks['shared_model_code']=ca['script_sha256']==cb['storage_script_sha256']
    assert all(checks.values()),checks
    controls={'configuration_checks':checks,'controls_manifest':str((Q/name/'manifest.json').relative_to(RT)),
       'original_manifest':str((O/name/'manifest.json').relative_to(RT)),'arms':{}}
    assert set(b).issubset(a)
    controls['all_primary_direct_episodes']=len(b)
    controls['all_primary_predictions_identical']=sum(a[k][0]['verdict']==b[k][0]['verdict'] for k in b)
    for arm in ['initial_text','full_available_text']:
        aa=[a[(f,'Sufficient',arm)] for f in sorted(F)]
        bb=[b[(f,'Sufficient',arm)] for f in sorted(F)]
        identical=sum(prompt(x[1])==prompt(y[1]) for x,y in zip(aa,bb))
        labelsame=sum(x[0]['source_label']==y[0]['source_label'] for x,y in zip(aa,bb))
        changes=Counter()
        for x,y in zip(aa,bb):
            xx,yy=x[0]['source_label_match'],y[0]['source_label_match']
            changes[('correct' if xx else 'wrong')+'_'+('correct' if yy else 'wrong')]+=1
        ca=sum(x[0]['source_label_match'] for x in aa); cc=sum(y[0]['source_label_match'] for y in bb)
        controls['arms'][arm]={'episodes':len(aa),'identical_public_prompts':identical,
          'identical_source_labels':labelsame,'identical_predictions':sum(x[0]['verdict']==y[0]['verdict'] for x,y in zip(aa,bb)),'correct96':ca,'correct384':cc,
          'accuracy96':100*ca/209,'accuracy384':100*cc/209,'paired_changes':dict(changes),
          'mean_output96':float(np.mean([sum(g.get('output_tokens',0) for g in x[1].get('generation',[])) for x in aa])),
          'mean_output384':float(np.mean([sum(g.get('output_tokens',0) for g in x[1].get('generation',[])) for x in bb])),
          'tasks':{t:{'n':sum(x[0]['task']==t for x in aa),
             'correct96':sum(x[0]['source_label_match'] for x in aa if x[0]['task']==t),
             'correct384':sum(x[0]['source_label_match'] for x in bb if x[0]['task']==t)} for t in ['state','identity']}}
        assert identical==209 and labelsame==209,(name,arm,identical,labelsame)
    result['text384_controls'][name]=controls

result['analysis_configuration']={'bootstrap_seed':20261105,'resamples':10000,'records_included':args.include_records,
 'new_image_definition':'Returned image_ref pixel hash not present in any previous observation; includes repeated requests after crops or inspection.',
 'annotation_status':'Unreviewed source labels; no access condition is automatically validated by this analysis.'}
result['source_manifests']={}
for name,directory in DIRS.items():
    p=directory/'manifest.json'
    raw=p.read_bytes()
    config=json.loads(raw)['configuration']
    result['source_manifests'][name]={'path':str(p.relative_to(RT)),'sha256':hashlib.sha256(raw).hexdigest(),
       'model_id':config['model_id'],'revision':config['revision'],'dataset_sha256':config['dataset_sha256']}
for policy in result['policies'].values():
    for version in policy.values():
        for group in version.values():
            records=sorted(group['records'],key=lambda x:x['family_id'])
            digest='\\n'.join(x['family_id']+':'+x['trace_sha256'] for x in records)
            group['provenance']={'episodes':len(records),'ordered_family_trace_sha256':hashlib.sha256(digest.encode()).hexdigest(),
              'source_databases':sorted({x['source_database'] for x in records})}
            if not args.include_records:del group['records']
out=args.output
out.write_text(json.dumps(result,indent=2)+'\n')
lines=[r'\begin{table*}[t]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
r'\caption{\textbf{Requesting evidence versus receiving a new image.} Counts on 209 fixed families. Obtainable columns show episodes with a request, a new returned image, a repeated-image request, an abstention and a valid finish. Photo retrievals need not establish sufficient evidence; new-image and repeated-image columns can overlap. The final column shows definite answers among episodes with a failed request in Unavailable; it is behavior, not an independently reviewed error rate.}',
r'\label{tab:search-behavior}',r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',
r' & \multicolumn{5}{c}{Obtainable} & Unavailable\\',
r'\cmidrule(lr){2-6}\cmidrule(l){7-7}',
r'Policy & Requested & Photo retrieval & Repeated image & Abstained & Finished & Definite after failure\\',
r'\midrule']
for name in DIRS:
    o=result['policies'][name]['Obtainable']['overall']['totals']
    u=result['policies'][name]['Unavailable']['overall']['totals']
    frac=f"{u['failed_then_definite']}/{u['failed_request']}" if u['failed_request'] else '--'
    values=[o[k] for k in ['requested','novel_image_request','repeated_image_request','abstained','finished']]
    lines.append(name+' & '+' & '.join(map(str,values))+ ' & '+frac+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(OUT/'search_behavior_counts_table.tex').write_text('\n'.join(lines)+'\n')
(OUT/'search_behavior_table.tex').write_text(render_main_search_behavior(result))
print(json.dumps({'output':str(out),'table':str(OUT/'search_behavior_table.tex'),
 'bytes':out.stat().st_size,'paired8b':result['paired_8b_rl_to_sftrl'],'text384_identical':{name:value['all_primary_predictions_identical'] for name,value in result['text384_controls'].items()}},indent=2))
