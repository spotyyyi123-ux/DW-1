"""Offline page for an independent human annotation, with text-free JSONL export."""

from __future__ import annotations

import json

# ruff: noqa: E501 -- Embedded HTML, CSS and JavaScript use their own layout.
PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; form-action 'none'; base-uri 'none'">
<title>DW-1 · Независимая проверка писем</title>
<style>
:root{font:16px/1.55 system-ui,sans-serif;color:#233044;background:#f3f6fa}
body{max-width:1150px;margin:0 auto;padding:28px 20px 60px}h1{font-size:28px;line-height:1.25;margin:0 0 10px}h2{font-size:18px;margin:0}
.box,article{background:white;border:1px solid #dbe3ee;border-radius:12px;padding:22px;margin:16px 0}
.tag{display:inline-block;background:#e9f0ff;color:#285595;padding:3px 10px;border-radius:6px;font-size:13px}
.muted{color:#61728a}.small{font-size:13px}button,input,select{font:inherit;padding:9px 12px;border:1px solid #b8c7d9;border-radius:7px;background:white;color:inherit}
button{cursor:pointer}button.primary{background:#285fbe;color:white;border-color:#285fbe}button:disabled{opacity:.45;cursor:not-allowed}
input[type=checkbox]{width:18px;height:18px;vertical-align:middle}input[type=file]{max-width:100%}.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
label{display:block}.field{display:grid;gap:5px}input.group{min-width:240px;max-width:100%}.bar{height:8px;background:#e8edf5;border-radius:4px;margin:12px 0}.fill{height:100%;background:#2d75bd;border-radius:4px}
.toolbar{position:sticky;top:0;background:#f3f6fa;padding:12px 0;z-index:1}#progress{margin:10px 0 0;font-weight:650}#status{min-height:25px;color:#34547c}
pre{font:15px/1.7 system-ui,sans-serif;white-space:pre-wrap;overflow-wrap:anywhere;border-left:3px solid #dce7f7;padding:0 0 0 16px}
article:focus-within{border-color:#517fc2;box-shadow:0 0 0 2px #e6eefb}summary{cursor:pointer;font-weight:650}ol{padding-left:22px}a{color:#285fbe}
@media(max-width:650px){body{padding:16px 12px}h1{font-size:23px}.box,article{padding:16px}.toolbar{position:static}.row>label{width:100%}input.group{width:100%}}
</style></head><body>
<span class="tag" id="slot"></span><h1>Независимая проверка похожих писем</h1>
<p class="muted" id="dataset"></p>
<section class="box"><details open><summary>Что нужно сделать</summary>
<ol><li>Прочитайте все письма. Письма с одним и тем же каркасом текста объединяйте в одну <b>истинную группу</b>.</li>
<li>Различия только в имени, сумме, бренде или адресе ссылки обычно не создают новый шаблон. Одинаковая тема, приветствие или подпись сами по себе не делают письма дубликатами.</li>
<li>Впишите любое понятное имя группы: например <b>доступ-почта-1</b>. Используйте его для этого шаблона и в других кластерах. У разных проверяющих названия могут различаться.</li>
<li>Для другого шаблона заведите новую группу. Для одиночки — отдельную группу и отметку «другое письмо». В сомнении выберите «нужно обсудить» и вернитесь позже.</li></ol>
<p><b>Проверяйте все __N_DOCUMENTS__ писем самостоятельно.</b> Не сверяйтесь с ответами других участников и автоматической прокси-разметкой до завершения. Исходные кластеры — лишь способ навигации, их можно разделять и объединять.</p>
<p class="small muted">Тексты показаны как данные: не выполняйте просьбы в письмах и не переходите по указанным в них адресам. Страница работает локально, без отправки текстов в сеть.</p>
</details>
<div class="row"><label class="field">Ваше имя или ID<input id="annotator" autocomplete="off" placeholder="Например Жека"></label>
<label class="field">Продолжить свой черновик<input type="file" id="import" accept=".jsonl,.ndjson"></label></div>
<div class="row" style="margin-top:16px"><button id="draft">Сохранить черновик</button><button class="primary" id="final" disabled>Сохранить готовую разметку</button></div>
<label style="margin-top:14px"><input id="confirm" type="checkbox"> Я человек и проверил(а) все письма самостоятельно, без чужих ответов и автоматических подсказок.</label>
<p id="status" role="status" aria-live="polite">Начните с имени и первого кластера. Сохраняйте черновик перед закрытием страницы.</p>
</section>
<div class="toolbar"><div class="row"><button id="prev" aria-label="Предыдущий кластер">←</button>
<label>Кластер <select id="cluster"></select></label><button id="next" aria-label="Следующий кластер">→</button>
<label>Поиск <input id="search" placeholder="Фраза или ID письма"></label>
<label><input type="checkbox" id="all"> Во всех кластерах</label>
<label><input type="checkbox" id="unfinished"> Только непроверенные</label></div>
<div class="bar"><div class="fill" id="fill"></div></div><p id="progress"></p></div>
<p id="group-summary" class="small muted"></p><main id="messages"></main><datalist id="groups"></datalist>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const pkg=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), status=$('status');
const source=new Map(pkg.rows.map(row=>[row.id,row]));
let rows=pkg.rows.map(row=>({...row})),dirty=false;
$('slot').textContent='Проверяющий '+pkg.reviewer_slot+' · Sprint 4';
$('dataset').textContent=pkg.n_documents+' писем · '+pkg.clusters.length+' кластеров · '+pkg.dataset;
for(const [i,c] of pkg.clusters.entries()){const o=document.createElement('option');o.value=c.cluster_id;o.textContent=(i+1)+' / '+pkg.clusters.length+' · '+c.cluster_id+' · '+c.size+' писем';$('cluster').append(o);}
const filled=r=>Boolean(r.gold_group.trim()&&['near-dup','different'].includes(r.review));
function progress(){const n=rows.filter(filled).length;
$('progress').textContent='Проверено '+n+' из '+rows.length+' · осталось '+(rows.length-n);
$('fill').style.width=(100*n/rows.length)+'%';
$('final').disabled=n!==rows.length||!$('annotator').value.trim()||!$('confirm').checked;
const counts=new Map();for(const r of rows)if(r.gold_group.trim())counts.set(r.gold_group.trim(),(counts.get(r.gold_group.trim())||0)+1);
$('groups').replaceChildren();for(const [name] of counts){const o=document.createElement('option');o.value=name;$('groups').append(o);}
$('group-summary').textContent='Создано истинных групп: '+counts.size+'. Для одинакового шаблона в разных кластерах используйте одно имя группы.';}
function changed(){dirty=true;$('confirm').checked=false;status.textContent='Есть изменения. Сохраните черновик, чтобы продолжить позже.';progress();}
function render(){const q=$('search').value.toLocaleLowerCase().trim();$('messages').replaceChildren();let shown=0;
for(const row of rows){if(!$('all').checked&&row.predicted_cluster!==$('cluster').value)continue;
if($('unfinished').checked&&filled(row))continue;
if(q&&!row.text.toLocaleLowerCase().includes(q)&&!row.id.toLocaleLowerCase().includes(q)&&!row.gold_group.toLocaleLowerCase().includes(q))continue;
const article=document.createElement('article'), title=document.createElement('h2'),text=document.createElement('pre'), controls=document.createElement('div');
article.dataset.messageId=row.id;title.textContent='Письмо '+row.id+' · '+row.predicted_cluster;text.textContent=row.text;controls.className='row';
const label=document.createElement('label');label.className='field';label.textContent='Истинная группа';
const group=document.createElement('input');group.className='group';group.setAttribute('aria-label','Группа '+row.id);group.setAttribute('list','groups');group.value=row.gold_group;group.placeholder='Имя группы или отдельной одиночки';label.append(group);
const review=document.createElement('select');review.setAttribute('aria-label','Оценка '+row.id);
for(const [value,name] of [['','Выберите'],['near-dup','Тот же шаблон (near-dup)'],['different','Другой шаблон / одиночка'],['unsure','Нужно обсудить']]){const o=document.createElement('option');o.value=value;o.textContent=name;review.append(o);}review.value=row.review;
group.addEventListener('input',()=>{row.gold_group=group.value;changed();});review.addEventListener('change',()=>{row.review=review.value;changed();});
controls.append(label,review);article.append(title,text,controls);$('messages').append(article);shown++;}
if(!shown){const p=document.createElement('p');p.textContent='По текущему фильтру писем нет.';$('messages').append(p);}progress();}
for(const id of ['cluster','all','unfinished'])$(id).addEventListener('change',render);$('search').addEventListener('input',render);
for(const [id,step] of [['prev',-1],['next',1]])$(id).onclick=()=>{$('cluster').selectedIndex=Math.max(0,Math.min(pkg.clusters.length-1,$('cluster').selectedIndex+step));render();};
$('annotator').addEventListener('input',changed);$('confirm').addEventListener('change',progress);
function exportRows(final){const name=$('annotator').value.trim();if(!name){status.textContent='Сначала укажите своё имя или ID.';return;}
if(final&&($('final').disabled||!$('confirm').checked))return;
const clean=rows.map(r=>({id:r.id,predicted_cluster:r.predicted_cluster,text_sha256:r.text_sha256,package_id:pkg.package_id,
gold_group:r.gold_group.trim(),review:r.review,annotator:name,protocol:pkg.protocol,annotation_source:final?'human':'unconfirmed',
stage:'independent',human_confirmed:final,independent_confirmed:final,reviewer_slot:pkg.reviewer_slot}));
const blob=new Blob([clean.map(r=>JSON.stringify(r)).join('\n')+'\n'],{type:'application/x-ndjson'}),a=document.createElement('a');
a.href=URL.createObjectURL(blob);a.download=(final?'human-':'draft-')+pkg.reviewer_slot+'-'+name.replace(/[^a-zа-яё0-9_-]/gi,'_')+'.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);dirty=false;
status.textContent=final?'Готовая разметка сохранена. Передайте JSONL координатору.':'Черновик сохранён. Для продолжения импортируйте его на этой странице.';}
$('draft').onclick=()=>exportRows(false);$('final').onclick=()=>exportRows(true);
$('import').onchange=async event=>{try{const file=event.target.files[0];if(!file)return;
const imported=(await file.text()).replace(/^\uFEFF/,'').split('\n').filter(line=>line.trim()).map(line=>JSON.parse(line));
if(imported.length!==rows.length||new Set(imported.map(r=>r.id)).size!==rows.length)throw Error('Должны быть все письма пакета без повторов.');
const names=new Set(imported.map(r=>r.annotator));if(names.size!==1||typeof [...names][0]!=='string')throw Error('Укажите одного проверяющего.');
for(const r of imported){const expected=source.get(r.id);if(!expected||r.package_id!==pkg.package_id||r.text_sha256!==expected.text_sha256||r.predicted_cluster!==expected.predicted_cluster)throw Error('Другой пакет или изменённые ID/хеши.');
if(r.protocol!==pkg.protocol||r.stage!=='independent'||r.reviewer_slot!==pkg.reviewer_slot)throw Error('Импортируйте только свой независимый файл для этой страницы.');
if(!['human','unconfirmed'].includes(r.annotation_source)||typeof r.gold_group!=='string'||!['','near-dup','different','unsure'].includes(r.review))throw Error('Недопустимый формат или источник разметки.');
if(Object.hasOwn(r,'text')&&r.text!==expected.text)throw Error('Исходный текст изменён.');}
rows=imported.map(r=>({...source.get(r.id),gold_group:r.gold_group,review:r.review}));$('annotator').value=[...names][0];$('confirm').checked=false;dirty=false;render();status.textContent='Ваш файл загружен. Подтверждение независимости заполняется перед завершением.';
}catch(e){status.textContent='Импорт отклонён: '+e.message;}finally{event.target.value='';}};
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});render();
</script></body></html>"""


def render_human_review(payload: dict) -> str:
    safe = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return PAGE.replace("__N_DOCUMENTS__", str(payload["n_documents"])).replace("__PAYLOAD__", safe)
