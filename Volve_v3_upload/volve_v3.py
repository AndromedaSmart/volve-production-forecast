"""Volve v3: direct horizons, nested selection, history-only vs oracle.
Run: python volve_v3.py --data data/data.xlsx --out reports/v3
Legacy v1/v2 modules are intentionally unchanged.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WINDOWS=[(1,'2009-12-01'),(2,'2011-12-01'),(3,'2013-12-01'),(4,'2014-12-01'),(5,'2015-12-01')]
CANDIDATES=[('mean6',None),('median6',None),('last',None)]+[(m,a) for m in ('ridge','ridge_wnf') for a in (1.,10.,100.)]
LABELS={'auto_inner':'Автовыбор','mean6':'Среднее 6 мес.','median6':'Медиана 6 мес.','ridge_a10':'Ridge alpha=10','ridge_wnf_a10':'Ridge + log1p(вода/нефть)','well_auto':'Скважинный автовыбор','well_mean6':'Скважинное среднее','well_median6':'Скважинная медиана','rate6_median_hours':'Дебит × медиана часов','rate6_oracle':'Дебит × факт. часы (oracle)'}
FL={'oil':'Нефть','gas':'Газ'}

def wape(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    return float(100*np.abs(y-p).sum()/np.abs(y).sum()) if np.isfinite(y).all() and np.isfinite(p).all() and np.abs(y).sum()>0 else np.nan

def load_data(path):
    m=pd.read_excel(path,sheet_name='Monthly Production Data',header=None).iloc[2:].copy()
    m.columns=['WELL','NPD_CODE','YEAR','MONTH','hours','oil','gas','water','gi','wi']
    for c in m.columns[1:]: m[c]=pd.to_numeric(m[c],errors='coerce')
    m=m.dropna(subset=['YEAR','MONTH']);m['WELL']=m.WELL.str.strip()
    m['DATE']=pd.to_datetime(dict(year=m.YEAR.astype(int),month=m.MONTH.astype(int),day=1))
    if m.duplicated(['WELL','DATE']).any(): raise ValueError('Duplicate well-month')
    for c in ('oil','gas','water'): m[c]=m[c].fillna(0.)
    idx=pd.date_range(m.DATE.min(),m.DATE.max(),freq='MS')
    wells={w:g.set_index('DATE')[['oil','gas','water','hours']].sort_index().reindex(idx) for w,g in m.groupby('WELL') if (g.oil+g.gas).sum()>0}
    field=m.groupby('DATE')[['oil','gas','water']].sum().reindex(idx)
    hours=m.loc[m.oil+m.gas>0].groupby('DATE').hours.sum()
    field['hours']=hours.reindex(idx,fill_value=0.)
    return m,field,wells

def trim(df):
    active=(df.oil.fillna(0)+df.gas.fillna(0))>0
    return df.loc[active[active].index[0]:].copy() if active.any() else df.iloc[:0].copy()

def features(df,o,fluid,physical=False):
    past=df.iloc[:o+1];y=past[fluid].to_numpy(float)
    if len(y)<6 or not np.isfinite(y[-6:]).all(): return None
    scale=max(float(np.mean(np.abs(y[-6:]))),1.)
    x=[y[-k]/scale for k in (1,2,3,6)]+[np.mean(y[-3:])/scale,np.mean(y[-6:])/scale,np.std(y[-6:])/scale]
    if physical:
        oil=past.oil.to_numpy(float)[-6:];water=past.water.to_numpy(float)[-6:]
        valid=(oil>0)&(water>=0)&np.isfinite(oil)&np.isfinite(water)
        if not valid.any(): return None
        ratio=np.log1p(water[valid]/oil[valid]);x += [float(ratio[-1]),float(ratio.mean())]
    return np.asarray(x,float),scale

def samples(df,fluid,h,physical=False):
    X,Y,meta=[],[],[]
    for o in range(5,len(df)-h):
        f=features(df,o,fluid,physical);y=df[fluid].iloc[o+h]
        if f is None or not np.isfinite(y): continue
        x,s=f;X.append(x);Y.append(y/s);meta.append((o,o+h,s))
    return np.asarray(X),np.asarray(Y),meta

def predict(df,fluid,cfg,diagnostics=False):
    name,alpha=cfg;y=df[fluid].to_numpy(float);tr=[]
    if name in ('mean6','median6','last'):
        rule={'mean6':np.mean,'median6':np.median,'last':lambda x:x[-1]}[name]
        if len(y)<6 or not np.isfinite(y[-6:]).all(): raise ValueError('Missing recent history')
        if diagnostics:
            for h in range(1,7):
                for o in range(5,len(df)-h):
                    tail=y[o-5:o+1]
                    if np.isfinite(tail).all() and np.isfinite(y[o+h]):
                        tr.append(dict(origin=df.index[o],DATE=df.index[o+h],horizon=h,actual=y[o+h],pred=max(0.,float(rule(tail)))))
        return np.full(6,max(0.,float(rule(y[-6:])))),tr
    f=features(df,len(df)-1,fluid,name=='ridge_wnf')
    if f is None: raise ValueError('Insufficient features')
    xx,scale=f;pred=[]
    for h in range(1,7):
        X,Y,meta=samples(df,fluid,h,name=='ridge_wnf')
        if len(Y)<6: raise ValueError('Too few training pairs')
        reg=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(X,Y)
        pred.append(max(0.,float(reg.predict(xx[None])[0])*scale))
        if diagnostics:
            fitted=np.maximum(reg.predict(X),0.)
            for z,(o,t,s) in enumerate(meta): tr.append(dict(origin=df.index[o],DATE=df.index[t],horizon=h,actual=Y[z]*s,pred=fitted[z]*s))
    return np.asarray(pred),tr

def tuning(df,fluid):
    rows=[]
    for cfg in CANDIDATES:
        scores=[]
        for end in (len(df)-6,len(df)-12,len(df)-18):
            if end<18: continue
            try:
                p,_=predict(df.iloc[:end],fluid,cfg);score=wape(df[fluid].iloc[end:end+6],p)
                if np.isfinite(score): scores.append(score)
            except ValueError: pass
        rows.append(dict(model=cfg[0],alpha=cfg[1],inner_wape=float(np.mean(scores)) if scores else np.nan,n_folds=len(scores)))
    eligible=[(r['inner_wape'],i) for i,r in enumerate(rows) if np.isfinite(r['inner_wape'])]
    return CANDIDATES[min(eligible)[1]] if eligible else ('mean6',None),rows

def run_window(field,wells,window,cutoff):
    cutoff=pd.Timestamp(cutoff);dates=pd.date_range(cutoff+pd.offsets.MonthBegin(),periods=6,freq='MS')
    hist=trim(field.loc[:cutoff]);fut=field.reindex(dates)
    rec=[];train=[];grids=[];choices=[];coverage=[];wellrows=[]
    def save(fluid,scope,scenario,model,p):
        for h,(d,a,b) in enumerate(zip(dates,fut[fluid],p),1): rec.append(dict(window=window,fluid=fluid,scope=scope,scenario=scenario,model=model,horizon=h,DATE=d,actual=float(a),pred=float(b)))
    def choose(df,fluid,scope):
        cfg,grid=tuning(df,fluid)
        grids.extend(dict(window=window,fluid=fluid,scope=scope,**r) for r in grid)
        row=next(r for r in grid if r['model']==cfg[0] and (cfg[1] is None or r['alpha']==cfg[1]))
        choices.append(dict(window=window,fluid=fluid,scope=scope,selected=cfg[0],alpha=cfg[1],inner_wape=row['inner_wape'],n_folds=row['n_folds'],history_months=len(df),fallback=row['n_folds']==0))
        return cfg
    for fluid in ('oil','gas'):
        cfg=choose(hist,fluid,'field')
        for label,c in [('auto_inner',cfg),('mean6',('mean6',None)),('median6',('median6',None)),('ridge_a10',('ridge',10.)),('ridge_wnf_a10',('ridge_wnf',10.))]:
            p,tr=predict(hist,fluid,c,True);save(fluid,'field','history_only',label,p)
            train.extend(dict(window=window,fluid=fluid,scope='field',model=label,**r) for r in tr)
        sums={k:np.zeros(6) for k in ('well_auto','well_mean6','well_median6','rate6_oracle','rate6_median_hours')}
        for well,df in wells.items():
            wh=trim(df.loc[:cutoff]);included=len(wh)>=6 and np.isfinite(wh[fluid].iloc[-6:]).all()
            actual=df[fluid].reindex(dates).fillna(0.).to_numpy(float)
            coverage.append(dict(window=window,fluid=fluid,WELL=well,status='included' if included else 'unseen_or_insufficient_history',test_volume=float(actual.sum())))
            if not included: continue
            wc=choose(wh,fluid,well)
            for label,c in [('well_auto',wc),('well_mean6',('mean6',None)),('well_median6',('median6',None))]:
                p,_=predict(wh,fluid,c);sums[label]+=p
                for h,(d,a,b) in enumerate(zip(dates,actual,p),1): wellrows.append(dict(window=window,fluid=fluid,WELL=well,model=label,horizon=h,DATE=d,actual=a,pred=b))
            recent=wh.iloc[-6:];ok=(recent.hours>0)&recent[fluid].notna()
            rate=float(recent.loc[ok,fluid].sum()/recent.loc[ok,'hours'].sum()) if ok.any() else 0.
            sums['rate6_oracle']+=rate*df.hours.reindex(dates).to_numpy(float)
            sums['rate6_median_hours']+=rate*float(recent.hours.median())
        for label,p in sums.items(): save(fluid,'known_wells','oracle_actual_hours' if label=='rate6_oracle' else 'history_only',label,p)
    return dict(predictions=rec,train_fitted=train,inner_validation=grids,model_choices=choices,coverage=coverage,well_predictions=wellrows)

def summarize(pr):
    keys=['window','fluid','scope','scenario','model'];rows=[]
    for key,g in pr.groupby(keys):
        a,p=g.actual.to_numpy(float),g.pred.to_numpy(float)
        rows.append(dict(zip(keys,key),WAPE=wape(a,p),MAE=float(np.mean(np.abs(a-p))),RMSE=float(np.sqrt(np.mean((a-p)**2))),n_valid=int((np.isfinite(a)&np.isfinite(p)).sum())))
    met=pd.DataFrame(rows)
    summ=met.groupby(keys[1:]).WAPE.agg(lambda x:float(np.sqrt(np.mean(x.to_numpy()**2))) if len(x)==5 and x.notna().all() else np.nan).rename('RMS_WAPE').reset_index()
    return met,summ

def artifacts(frames,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from docx import Document
    from docx.shared import Inches
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    pr=frames['predictions'];tr=frames['train_fitted'];met=frames['window_metrics'];summ=frames['summary']
    pics=dest/'figures';pics.mkdir(exist_ok=True);manifest=[]
    def save(fig,key,title,caption):
        fig.tight_layout();fig.savefig(pics/(key+'.png'),dpi=145,bbox_inches='tight');plt.close(fig)
        manifest.append(dict(id=key,title=title,caption=caption,file=key+'.png'))
    for w in range(1,6):
        for fl in ('oil','gas'):
            scale=1e3 if fl=='oil' else 1e6;unit='тыс. ст. м³/мес.' if fl=='oil' else 'млн ст. м³/мес.'
            g=pr.loc[(pr.window==w)&(pr.fluid==fl)]
            fig,axs=plt.subplots(3,1,figsize=(12,7.5),sharex=True)
            a=g.groupby('DATE').actual.first();axs[0].plot(a.index,a/scale,'ko-',label='Факт')
            for name in ('auto_inner','mean6','median6','well_median6'):
                k=g.loc[g.model==name].sort_values('DATE')
                for ax,col,s in zip(axs,('pred','absolute_error','absolute_relative_error_pct'),(scale,scale,1)):
                    ax.plot(k.DATE,k[col]/s,'o--',label=LABELS[name])
            title=f'{FL[fl]}: окно {w}, прогноз и ошибки';axs[0].set_title(title);axs[0].legend(fontsize=8,ncol=2)
            for ax,label in zip(axs,(unit,'Абсолютная ошибка, '+unit,'Абс. относительная ошибка, %')): ax.set_ylabel(label);ax.grid(alpha=.2)
            fig.autofmt_xdate();save(fig,f'T{w}_{fl}',title,'Только история. Скважинная медиана — фиксированный кандидат, не независимо подтверждённый победитель.')
            tg=tr.loc[(tr.window==w)&(tr.fluid==fl)&(tr.model=='auto_inner')]
            fig,axs=plt.subplots(3,1,figsize=(12,7.5),sharex=True)
            for h,k in tg.groupby('horizon'):
                for ax,col,s in zip(axs,('error','absolute_error','absolute_relative_error_pct'),(scale,scale,1)): ax.plot(k.DATE,k[col]/s,label=f'h={h}',alpha=.7)
            title=f'{FL[fl]}: окно {w}, обучающий период';axs[0].set_title(title);axs[0].legend(ncol=6)
            for ax,label in zip(axs,('Знаковая ошибка, '+unit,'Абсолютная ошибка, '+unit,'Абс. относительная ошибка, %')): ax.set_ylabel(label);ax.grid(alpha=.2)
            fig.autofmt_xdate();save(fig,f'R{w}_{fl}',title,'Ridge: in-sample подгонка на всей обучающей истории. Для среднего/медианы — исторические ошибки правила. Не оценка будущей точности. Линии — горизонты 1…6.')
    order=['auto_inner','mean6','median6','ridge_a10','ridge_wnf_a10','well_auto','well_median6','rate6_median_hours']
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    for ax,fl in zip(axs,('oil','gas')):
        z=summ.loc[(summ.fluid==fl)&(summ.scenario=='history_only')].set_index('model').reindex(order)
        ax.barh(range(len(z)),z.RMS_WAPE)
        ax.set_yticks(range(len(z)),[LABELS[k] for k in order],fontsize=8);ax.invert_yaxis();ax.set_title(FL[fl]);ax.set_xlabel('RMS WAPE, %')
        for i,v in enumerate(z.RMS_WAPE): ax.text(v+.4,i,f'{v:.1f}%',va='center')
    save(fig,'S1_summary','Сравнение методов','RMS WAPE по пяти окнам; oracle исключён. Пять окон — ретроспективная проверка, не новый holdout.')
    for fl in ('oil','gas'):
        z=met.loc[(met.fluid==fl)&(met.scenario=='history_only')].pivot(index='model',columns='window',values='WAPE').reindex(order)
        fig,ax=plt.subplots(figsize=(12,5));im=ax.imshow(z,aspect='auto',cmap='YlOrRd',vmin=0,vmax=100)
        ax.set_yticks(range(len(z)),[LABELS[k] for k in order]);ax.set_xticks(range(5),range(1,6));ax.set_xlabel('Окно');ax.set_title(FL[fl])
        for i in range(len(z)):
            for j in range(5): ax.text(j,i,f'{z.iloc[i,j]:.1f}',ha='center',va='center')
        fig.colorbar(im,ax=ax);save(fig,'S2_'+fl,FL[fl]+': WAPE по окнам','Числа — WAPE, %. Цвет ограничен 100%, значения могут быть выше.')
    groups=[('S3_physical',['ridge_a10','ridge_wnf_a10','auto_inner'],'Проверка физического признака'),('S4_wells',['auto_inner','median6','well_auto','well_median6'],'Скважинное сравнение'),('S7_hours',['auto_inner','rate6_median_hours','rate6_oracle'],'История и oracle с фактическими часами')]
    for key,names,title in groups:
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        for ax,fl in zip(axs,('oil','gas')):
            for name in names:
                k=met.loc[(met.fluid==fl)&(met.model==name)].sort_values('window');ax.plot(k.window,k.WAPE,'o-',label=LABELS[name])
            ax.set_title(FL[fl]);ax.set_xlabel('Окно');ax.set_ylabel('WAPE, %');ax.legend(fontsize=7);ax.grid(alpha=.2)
        save(fig,key,title,'Oracle — не настоящий план. WAPE окна 5 oracle не определён из-за отсутствия часов F-1 C за май–июнь 2016. Физический признак нестабилен; универсальный выигрыш не подтверждён.')
    cov=frames['coverage'];cv=[]
    for (w,fl),g in cov.groupby(['window','fluid']): cv.append(dict(window=w,fluid=fl,uncovered_pct=100*g.loc[g.status!='included','test_volume'].sum()/g.test_volume.sum()))
    cv=pd.DataFrame(cv);frames['coverage_summary']=cv
    fig,ax=plt.subplots(figsize=(11,4))
    for fl,shift in [('oil',-.18),('gas',.18)]:
        k=cv.loc[cv.fluid==fl];ax.bar(k.window+shift,k.uncovered_pct,width=.34,label=FL[fl])
    ax.legend();ax.set_xlabel('Окно');ax.set_ylabel('Доля факта вне скважинного прогноза, %')
    save(fig,'S5_coverage','Покрытие скважинного прогноза','Новые/коротко наблюдавшиеся стволы не добавляются задним числом. Это доля факта, а не доказанная доля итоговой ошибки.')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,fl in zip(axs,('oil','gas')):
        for name in ('auto_inner','mean6','median6','well_median6'):
            g=pr.loc[(pr.fluid==fl)&(pr.model==name)]
            vals=[wape(k.actual,k.pred) for h,k in g.groupby('horizon')];ax.plot(range(1,7),vals,'o-',label=LABELS[name])
        ax.set_title(FL[fl]);ax.set_xlabel('Горизонт, мес.');ax.set_ylabel('WAPE по пяти датам, %');ax.legend(fontsize=7)
    save(fig,'S6_horizons','Ошибка по горизонтам','WAPE отдельно по каждому горизонту — другая агрегация, не RMS WAPE по окнам.')
    (dest/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    doc=Document();doc.add_heading('Volve v3: выполненный расчёт',0)
    doc.add_paragraph('Прямые горизонты 1…6 без линейного времени и рекурсии. Основной сценарий — только история. Архивный план не предоставлен. Oracle с фактическими будущими часами отделён. Пять окон уже изучались при разработке: это ретроспективная оценка, не новый независимый тест.')
    doc.add_paragraph('Автовыбор использует внутренние отсечения −6/−12/−18 месяцев, минимум 18 месяцев истории. В первом окне 23 месяца: действует заранее заданный mean6. Кандидаты: mean6/median6/last/Ridge/Ridge+log1p(вода/нефть), alpha=1/10/100. Лаги 1/2/3/6, средние 3/6, стандартное отклонение 6. Масштаб каждой пары — средний абсолютный уровень шести месяцев на дате происхождения. StandardScaler обучается только на обучающих парах.')
    doc.add_paragraph('Физический признак — гипотеза log1p(вода/нефть), не буквальное ln(ВНФ). Скважинная сумма включает стволы с ≥6 известными месяцами и оценивается против полного факта. Пропуски часов F-1 C за май–июнь 2016 не заменяются нулями: oracle RMS по пяти окнам не определён. Арпс, LightGBM, Transformer, CSDI и нефть×ГФ из v2 не пересчитывались.')
    t=doc.add_table(rows=1,cols=3);t.style='Light Shading Accent 1'
    for c,v in zip(t.rows[0].cells,['Метод','Нефть RMS WAPE, %','Газ RMS WAPE, %']): c.text=v
    for name in order:
        row=t.add_row().cells;row[0].text=LABELS[name]
        for i,fl in enumerate(('oil','gas'),1): row[i].text=f'{summ.loc[(summ.fluid==fl)&(summ.model==name),"RMS_WAPE"].iloc[0]:.2f}'
    doc.add_paragraph('Автовыбор почти не улучшает медиану. Скважинная медиана ретроспективно лучше, но выбирать победителя по тому же тесту нельзя. Физический Ridge alpha=10 нестабилен. WAPE=100×Σ|факт−прогноз|/Σ|факт|; RMS WAPE=sqrt(mean(WAPE²)) по пяти окнам. При нулевом факте месячная относительная ошибка не определяется. Интервалы неопределённости не оценивались.')
    for item in sorted(manifest,key=lambda x:x['id']):
        doc.add_page_break();doc.add_heading(item['id']+': '+item['title'],1);doc.add_picture(str(pics/item['file']),width=Inches(6.3));doc.add_paragraph(item['caption'])
    doc.save(dest/'Volve_v3_report.docx')
    with pd.ExcelWriter(dest/'Volve_v3_metrics.xlsx',engine='openpyxl') as writer:
        for name in ('summary','window_metrics','model_choices','coverage','coverage_summary','predictions','inner_validation','well_predictions'): frames[name].to_excel(writer,sheet_name=name[:31],index=False)
    return manifest

def run(path='data/data.xlsx',dest='reports/v3',render=True):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    monthly,field,wells=load_data(path);collected={}
    for w,cut in WINDOWS:
        for name,rows in run_window(field,wells,w,cut).items(): collected.setdefault(name,[]).extend(rows)
        print('Completed window',w,flush=True)
    frames={k:pd.DataFrame(v) for k,v in collected.items()}
    frames['window_metrics'],frames['summary']=summarize(frames['predictions'])
    for name in ('predictions','train_fitted'):
        df=frames[name];df['error']=df.pred-df.actual;df['absolute_error']=df.error.abs()
        df['absolute_relative_error_pct']=np.where(df.actual!=0,100*df.absolute_error/df.actual.abs(),np.nan)
    if render: artifacts(frames,dest)
    for name,df in frames.items(): df.to_csv(dest/(name+'.csv'),index=False)
    audit={'input_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'monthly_rows':len(monthly),'windows':5,'scenario':'history_only + separate oracle','archive_plan_available':False,'legacy_models_recomputed':False}
    (dest/'execution_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return frames

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/data.xlsx');p.add_argument('--out',default='reports/v3');a=p.parse_args()
    print(run(a.data,a.out)['summary'].to_string(index=False))
