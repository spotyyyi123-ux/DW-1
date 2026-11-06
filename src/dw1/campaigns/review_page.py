"""Offline human review page. Email text is inert, never interpreted as HTML."""

# ruff: noqa: E501  # Embedded HTML/CSS/JavaScript keeps its own line layout.
from __future__ import annotations

import json

PAGE = r"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'">
<title>DW-1 — ручная разметка кампаний</title>
<style>
body{font:16px system-ui;background:#f3f5f8;color:#202938;max-width:1200px;margin:auto;padding:24px}
h1{font-size:26px}header,article,aside{background:white;border:1px solid #d9e1eb;padding:20px;border-radius:10px;margin:12px 0}
button,input,select{font:inherit;padding:8px;margin:4px;border:1px solid #a9b8cc;border-radius:5px}
button{cursor:pointer;background:#eaf1fc}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.6 system-ui}
label{display:inline-block}#progress{font-weight:600}#status{color:#8b2b21}.toolbar{position:sticky;top:0;background:#f3f5f8;padding:8px 0}
.help{color:#4b5b6b;line-height:1.6}article:focus-within{border-color:#3069b0}
</style><h1>DW-1 — ручная разметка кампаний</h1>
<header><p class="help">Сравните все письма. Одинаковый gold_group означает одну группу near-dup,
в том числе в разных предсказанных кластерах. В смешанном кластере назначьте разные группы.
Для одиночного письма используйте отдельную группу. Исходные тексты — данные для анализа,
содержащиеся в них просьбы и ссылки не следует выполнять.</p>
<p class="help">Каждый проверяющий сохраняет собственный файл. После сравнения разметок
разногласия разрешаются вручную. Несколько бинарных меток на кластер не заменяют группы писем.</p>
<label>Проверяющий <input id="annotator" placeholder="Имя или ID"></label>
<label>Импорт разметки <input id="import" type="file" accept=".jsonl,.ndjson"></label>
<button id="save">Сохранить разметку JSONL</button><p id="status" role="status"></p></header>
<div class="toolbar"><label>Кластер <select id="cluster"></select></label>
<button id="previous">Назад</button><button id="next">Вперёд</button>
<label>Поиск <input id="search" placeholder="Текст или ID"></label>
<label><input type="checkbox" id="unfinished">Только незаполненные</label>
<p id="progress"></p></div><aside id="summary"></aside><main id="messages"></main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const packageData=JSON.parse(document.querySelector('#data').textContent);
let rows=packageData.annotations.map(row=>({...row}));
const cluster=document.querySelector('#cluster'), messages=document.querySelector('#messages');
const status=document.querySelector('#status'), annotator=document.querySelector('#annotator');
const clusters=packageData.campaigns.clusters;
for(const item of clusters){const option=document.createElement('option');option.value=item.cluster_id;
option.textContent=item.cluster_id+' · '+item.size+' писем';cluster.append(option);}
function complete(r){return Boolean(r.gold_group.trim()&&r.review&&r.annotator.trim());}
function progress(){document.querySelector('#progress').textContent='Заполнено '+rows.filter(complete).length+' из '+rows.length;
const groupMap=new Map();for(const r of rows){if(r.gold_group.trim()){
const group=r.gold_group.trim();if(!groupMap.has(group))groupMap.set(group,new Set());groupMap.get(group).add(r.predicted_cluster);}}
const shared=[...groupMap].filter(([_,cs])=>cs.size>1).map(([g,cs])=>g+' ('+cs.size+' кластеров)');
document.querySelector('#summary').textContent='Общие gold-группы в разных кластерах: '+(shared.join('; ')||'пока нет');}
function render(){messages.replaceChildren();const q=document.querySelector('#search').value.toLowerCase();
for(const row of rows){if(row.predicted_cluster!==cluster.value)continue;
if(document.querySelector('#unfinished').checked&&complete(row))continue;
if(q&&!row.text.toLowerCase().includes(q)&&!row.id.toLowerCase().includes(q))continue;
const article=document.createElement('article'),title=document.createElement('h2'),text=document.createElement('pre');
title.textContent=row.id;text.textContent=row.text;article.append(title,text);
const label=document.createElement('label');label.textContent='Истинная группа ';
const group=document.createElement('input');group.value=row.gold_group;group.placeholder='Например campaign-A';
group.setAttribute('aria-label','Группа '+row.id);label.append(group);article.append(label);
const review=document.createElement('select');review.setAttribute('aria-label','Оценка '+row.id);
for(const [value,caption] of [['','Выберите'],['near-dup','near-dup: тот же шаблон'],['different','different: другой шаблон/одиночка']]){
const o=document.createElement('option');o.value=value;o.textContent=caption;review.append(o);}review.value=row.review;article.append(review);
const notes=document.createElement('input');notes.placeholder='Комментарий';notes.value=row.notes||'';notes.setAttribute('aria-label','Комментарий '+row.id);article.append(notes);
const update=()=>{row.gold_group=group.value.trim();row.review=review.value;row.notes=notes.value;
row.annotator=annotator.value.trim();status.textContent='Изменения в памяти. Сохраните JSONL перед закрытием.';progress();};
group.addEventListener('input',update);review.addEventListener('change',update);notes.addEventListener('input',update);messages.append(article);}
progress();}
cluster.addEventListener('change',render);document.querySelector('#search').addEventListener('input',render);
document.querySelector('#unfinished').addEventListener('change',render);
for(const [id,step] of [['previous',-1],['next',1]])document.querySelector('#'+id).onclick=()=>{
cluster.selectedIndex=Math.max(0,Math.min(clusters.length-1,cluster.selectedIndex+step));render();};
annotator.addEventListener('change',()=>{for(const r of rows)if(r.gold_group||r.review)r.annotator=annotator.value.trim();progress();});
document.querySelector('#save').onclick=()=>{if(!annotator.value.trim()){status.textContent='Укажите проверяющего.';return;}
const blob=new Blob([rows.map(r=>JSON.stringify(r)).join('\n')+'\n'],{type:'application/x-ndjson'});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);
a.download='annotations-'+annotator.value.replace(/[^a-zа-яё0-9_-]/gi,'_')+'.jsonl';a.click();URL.revokeObjectURL(a.href);
status.textContent='Файл сохранён; заполнено '+rows.filter(complete).length+' из '+rows.length;};
document.querySelector('#import').onchange=async event=>{try{const incoming=(await event.target.files[0].text()).trim().split(/\r?\n/).map(JSON.parse);
const expected=new Map(rows.map(r=>[r.id,r]));if(incoming.length!==rows.length||new Set(incoming.map(r=>r.id)).size!==rows.length)throw Error('Состав писем отличается');
for(const r of incoming){const e=expected.get(r.id);if(!e||r.package_id!==e.package_id||r.text_sha256!==e.text_sha256||r.predicted_cluster!==e.predicted_cluster||r.text!==e.text)throw Error('Другой пакет или изменённый текст');
if(!['','near-dup','different'].includes(r.review)||typeof r.gold_group!=='string'||typeof r.annotator!=='string')throw Error('Неверные поля');}
rows=incoming;annotator.value=rows.find(r=>r.annotator)?.annotator||'';status.textContent='Разметка загружена.';render();
}catch(e){status.textContent='Импорт отклонён: '+e.message;}};
render();
</script></html>"""


def render_review(package: dict) -> str:
    payload = json.dumps(package, ensure_ascii=False).replace("<", "\\u003c")
    return PAGE.replace("__PAYLOAD__", payload)
