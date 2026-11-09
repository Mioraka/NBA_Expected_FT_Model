#!/usr/bin/env python3
"""Build player-season and distance-detail tables from saved source snapshots."""
import argparse
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ALIASES = {
 'enesfreedom':'eneskanter','nicclaxton':'nicolasclaxton',
 'charlesbrownjr':'charliebrownjr','juanhernangomez':'juanchohernangomez',
 'lucavildoza':'lukavildoza',
 'cameronreynolds':'camreynolds', 'cuiyongxi':'cuicui',
 'jeenathanwilliams':'natewilliams', 'kenyonmartinjr':'kjmartin',
 'rjnembhardjr':'rubennembhard', 'rubennembhardjr':'rubennembhard',
 'nigelhayes':'nigelhayesdavis', 'walterlemonjr':'waltlemonjr',
 'anferneehardaway':'pennyhardaway', 'michealrayrichardson':'michaelrayrichardson',
 'kristapsporziaaaais':'kristapsporzingis', 'darioaaaric':'dariosaric',
 'anaa34ejspasecaaiks':'anzejspasecniks', 'draenpetrovi':'drazenpetrovic',
 'aaarnasmariulionis':'sarunasmarciulionis',
}

def repair(s):
    for _ in range(3):
        candidates=[]
        for enc in ('cp1252','latin1'):
            try: candidates.append(s.encode(enc).decode('utf8'))
            except (UnicodeError,LookupError): pass
        if not candidates: break
        s=min(candidates,key=lambda z:sum(z.count(c) for c in ('Ã','Â','Ä','Å','â')))
    return s

def norm(s):
    s=repair(str(s))
    n=re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())
    return ALIASES.get(n,n)

def bare(s):
    return re.sub(r'(jr|iii|ii|iv|sr)$','',s)

def snapshot(path):
    if Path(path).suffix=='.parquet':return pd.read_parquet(path)
    v=json.loads(Path(path).read_text())['values']
    return pd.DataFrame([r+['']*(len(v[0])-len(r)) for r in v[1:]],columns=v[0]).replace('',np.nan)

def build(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    raw=snapshot(args.raw);awards=snapshot(args.awards)
    legacy=pd.read_excel(args.legacy)
    detail=pd.read_parquet(args.summary)
    for d,c in [(raw,'PLAYER_NAME'),(awards,'PLAYER_NAME'),(legacy,'player'),(detail,'Player')]: d['_name']=d[c].map(norm)
    # Alias resolution may drop a suffix only when there is one target identity name.
    allnames=set(raw['_name'])|set(legacy['_name'])|set(detail['_name'])
    bases={b:sorted(n for n in allnames if bare(n)==b) for b in {bare(n) for n in allnames}}
    def canonical(n):
        opts=bases.get(bare(n),[])
        return bare(n) if len(opts)>1 and all(bare(o)==bare(n) for o in opts) else n
    # Tim/Glen/Patrick and other father-son pairs must keep suffix identities separate.
    protected={'timhardaway','glenrice','patrickewing','larrynance','garypayton','garytrent','ronharper','larrydrew','johnlucas'}
    def canon(n): return n if bare(n) in protected else canonical(n)
    for d in (raw,awards,legacy,detail):d['_name']=d['_name'].map(canon)
    # Different players with the same name are separated by non-overlapping birth-year estimates.
    raw['_birth']=raw.SEASON_YEAR-raw.AGE
    clusters={}
    for n,g in raw.groupby('_name'):
        vals=sorted(g['_birth'].dropna().unique()); groups=[]
        for v in vals:
            if not groups or v-groups[-1][-1]>(0 if n=='marcuswilliams' else 1):groups.append([v])
            else:groups[-1].append(v)
        clusters[n]=groups
    def entity(r):
        groups=clusters[r['_name']]
        if len(groups)<=1:return r['_name']
        group=next(g for g in groups if r['_birth'] in g)
        return r['_name']+':birth'+str(int(min(group)))
    raw['player_key']=[entity(r) for _,r in raw.iterrows()]
    
    if raw.duplicated(['player_key','SEASON_YEAR']).any():
        raise ValueError(raw[raw.duplicated(['player_key','SEASON_YEAR'],False)][['PLAYER_NAME','SEASON_YEAR','AGE','TEAM','player_key']].to_string(index=False))
    raw['_row']=np.arange(len(raw))
    byname={n:g for n,g in raw.groupby('_name')}
    match_audit=[]
    byyear={(n,int(y)):g for (n,y),g in raw.groupby(['_name','SEASON_YEAR'])}
    def match(n,year,teams=None,kind=''):
        g=byname.get(n)
        if g is None:return None
        y=byyear.get((n,year),g.iloc[:0])
        if len(y)==1:return y.iloc[0].player_key
        if len(y)>1 and teams:
            toks=set(str(teams).split(','));z=y[y.TEAM.isin(toks)]
            if len(z)==1:return z.iloc[0].player_key
        if g.player_key.nunique()==1:return g.player_key.iloc[0]
        return None
    legacy['player_key']=[match(r['_name'],int(r.season),r.teams,'legacy') for _,r in legacy.iterrows()]
    # Map each modern NBA ID through unambiguous season/name/team matches.
    idkeys=legacy.dropna(subset=['player_key']).groupby('player_id').player_key.agg(lambda x:sorted(set(x)))
    assert all(len(x)==1 for x in idkeys), 'One NBA ID matched multiple identities'
    idmap={int(k):v[0] for k,v in idkeys.items()}
    detail_idmap={int(r['Player ID']):idmap.get(int(r['Player ID'])) or match(r['_name'],int(r['Season Code'])+1) for _,r in detail.drop_duplicates('Player ID').iterrows()}
    detail['player_key']=detail['Player ID'].map(detail_idmap)
    missing=detail[detail.player_key.isna()][['Player','Player ID']].drop_duplicates()
    detail.loc[detail.player_key.isna(),'player_key']=detail.loc[detail.player_key.isna(),'Player ID'].map(lambda x:f'nba:{int(x)}')
    raw['raw_stats_matched']=True
    existing=set(zip(raw.player_key,raw.SEASON_YEAR))
    extra=[]
    for _,r in detail[['player_key','Player','Season Code']].drop_duplicates().iterrows():
        year=int(r['Season Code'])+1
        if (r.player_key,year) not in existing:
            extra.append({'player_key':r.player_key,'PLAYER_NAME':r.Player,'SEASON_YEAR':year,'SEASON':f'{year-1}-{year}','Season_Player':f'{r.Player} - {year}','raw_stats_matched':False})
            existing.add((r.player_key,year))
    raw=pd.concat([raw,pd.DataFrame(extra)],ignore_index=True)
    # Career scorer qualification is based on complete supplied Raw Data history.
    scorer=set(raw.loc[(raw.GP>=60)&(raw.PTS>25),'player_key'])
    award_map={};season_awards={};unmatched_awards=[]
    for _,r in awards.iterrows():
        key=match(r['_name'],int(r.SEASON_YEAR))
        if key is None and r['_name'] in {'bobbyjones','eddiejohnson','georgejohnson'}:
            # Historical award rows belong to the older namesake.
            g=byname[r['_name']]
            key=g.sort_values('_birth').player_key.iloc[0]
        if key is None:
            if r['_name'] in byname:unmatched_awards.append({'name':r.PLAYER_NAME,'season':int(r.SEASON_YEAR)})
            continue
        vals={c:int(r[c]) for c in awards.columns if c not in ['SEASON_YEAR','SEASON','Season_Player','PLAYER_NAME','_name']}
        season_awards[(key,int(r.SEASON_YEAR))]=vals
        label=3 if vals['MVP'] else 2 if vals['ALL_NBA'] else 1 if vals['ALL_STAR'] else 0
        award_map[key]=max(award_map.get(key,0),label)
    labels={0:'None',1:'AS',2:'All-NBA',3:'MVP'}
    for d in (raw,detail):
        d['scorer_flag']=np.where(d.player_key.isin(scorer),'Yes','No')
        d['award_flag']=d.player_key.map(lambda k:labels[award_map.get(k,0)])
    # Reconstruct observed means within the exact summary bands from source FGA events.
    if args.bands:
        bands=pd.read_parquet(args.bands)
    else:
        aggs=[]
        for batch in pq.ParquetFile(args.events).iter_batches(batch_size=200000,columns=['pid','etype','opt1','de','locX','locY','_season','_season_type']):
            f=batch.to_pandas();f=f[f.etype.isin([1,2])&(f.pid>0)].copy()
            f['points']=np.where(f.opt1.isin([2,3]),f.opt1,np.where(f.de.str.lower().str.contains('3pt'),3,2))
            valid=f.locX.notna()&f.locY.notna()&~((f.locX==0)&(f.locY==-80))
            f['dist']=np.where(valid,np.hypot(f.locX,f.locY)/10,np.nan)
            f['band']=f.apply(lambda r:'Unknown' if pd.isna(r.dist) else ('Corner 3' if 22<=r.dist<=23.75 else 'Arc 3' if r.dist>23.75 else '3PT below 22') if r.points==3 else f'{int(np.floor(r.dist))} to <{int(np.floor(r.dist))+1}',axis=1)
            aggs.append(f.groupby(['pid','_season','_season_type','points','band'],dropna=False).dist.agg(['sum','count']).reset_index())
        bands=pd.concat(aggs).groupby(['pid','_season','_season_type','points','band'],as_index=False)[['sum','count']].sum()
    bands['band_mean']=bands['sum']/bands['count'].replace(0,np.nan)
    # FGA-weighted observed band means equal the pooled observed-shot mean.
    sums=bands.groupby(['pid','_season','_season_type'])[['sum','count']].sum()
    twos=bands[bands.points==2].groupby(['pid','_season','_season_type'])[['sum','count']].sum()
    means=pd.DataFrame({'AVG_Dist':sums['sum']/sums['count'].replace(0,np.nan),'2P_Dist':twos['sum']/twos['count'].replace(0,np.nan),'Distance_FGA':sums['count'],'Distance_2P_FGA':twos['count']}).reset_index()
    means=means.rename(columns={'pid':'Player ID','_season':'Season Code','_season_type':'Type'});means.Type=means.Type.map({'rg':'RS','po':'PO'})
    detail=detail.merge(means,on=['Player ID','Season Code','Type'],how='left',validate='many_to_one')
    ids=detail[['player_key','Player ID']].drop_duplicates();assert not ids.player_key.duplicated().any()
    raw=raw.merge(ids,on='player_key',how='left',validate='many_to_one')
    known=legacy.dropna(subset=['player_key'])[['player_key','player_id']].drop_duplicates()
    known=known.groupby('player_key').player_id.agg(lambda x: next(iter(x)) if len(set(x))==1 else np.nan)
    raw['Player ID']=raw['Player ID'].fillna(raw.player_key.map(known)).astype('Int64')
    legacycols=['FGA_2P','FGA_3P','FTA_Total','Foul_2P','Foul_2P1','FTA_2P','Foul_3P','Foul_3P1','FTA_3P','FTA_Other']
    leg=legacy.dropna(subset=['player_key']).copy()
    assert not leg.duplicated(['player_key','season']).any()
    raw=raw.merge(leg[['player_key','season']+legacycols],left_on=['player_key','SEASON_YEAR'],right_on=['player_key','season'],how='left',validate='one_to_one').drop(columns='season')
    rs=detail[detail.Type=='RS'][['player_key','Season Code','2P_Dist','AVG_Dist','Distance_FGA','Distance_2P_FGA']].drop_duplicates();rs['SEASON_YEAR']=rs['Season Code']+1
    raw=raw.merge(rs.drop(columns='Season Code'),on=['player_key','SEASON_YEAR'],how='left',validate='one_to_one')
    metrics=[c for c in detail.columns if c.startswith(('FGA ','FTA ')) or c in ['Shooting Fouls','FTM Linked']]
    totals=detail[detail.Type=='RS'].groupby(['player_key','Season Code'],as_index=False)[metrics].sum()
    totals['SEASON_YEAR']=totals['Season Code']+1
    totals=totals.drop(columns='Season Code').rename(columns={c:'pbp_'+c.replace(' ','_').replace('+','plus') for c in metrics})
    raw=raw.merge(totals,on=['player_key','SEASON_YEAR'],how='left',validate='one_to_one')
    for c in ['ALL_STAR','ALL_NBA_1','ALL_NBA_2','ALL_NBA_3','ALL_DEF_1','ALL_DEF_2','ALL_NBA','ALL_DEF','MVP','DPOY','FMVP']:
        raw['season_'+c]=[season_awards.get((k,int(y)),{}).get(c,0) for k,y in zip(raw.player_key,raw.SEASON_YEAR)]
    raw['legacy_stats_matched']=raw.FGA_2P.notna();raw['distance_stats_matched']=raw.Distance_FGA.notna()
    cross=raw[['PLAYER_NAME','SEASON_YEAR','TEAM','AGE','player_key','Player ID','legacy_stats_matched','distance_stats_matched']].sort_values(['player_key','SEASON_YEAR'])
    for d in (raw,detail):d.drop(columns=[c for c in d if c.startswith('_')],inplace=True)
    raw=raw.sort_values(['player_key','SEASON_YEAR']);detail=detail.sort_values(['player_key','Season Code','Type','Shot_Dist'])
    assert len(raw)==19968+len(extra) and len(detail)==120694
    assert not raw.duplicated(['player_key','SEASON_YEAR']).any()
    assert detail['FGA Total'].sum()==2018590 and detail['FTA Awarded Total'].sum()==402332
    assert detail.scorer_flag.notna().all() and detail.award_flag.notna().all()
    for name,d in [('player_season_stats_table',raw),('player_shooting_foul_details',detail)]:
        d.to_csv(out/(name+'.csv'),index=False);d.to_parquet(out/(name+'.parquet'),index=False)
    cross.to_csv(out/'player_match_audit.csv',index=False)
    bands.to_parquet(out/'observed_distance_band_means.parquet',index=False)
    audit={'season_rows':len(raw),'detail_rows':len(detail),'scorer_players':len(scorer),'award_player_counts':raw.drop_duplicates('player_key').award_flag.value_counts().to_dict(),'detail_unmatched_players':len(missing),'raw_legacy_matches':int(raw.legacy_stats_matched.sum()),'raw_distance_matches':int(raw.distance_stats_matched.sum()),'ambiguous_awards':unmatched_awards,'fga_total':int(detail['FGA Total'].sum()),'shooting_fta_total':int(detail['FTA Awarded Total'].sum())}
    (out/'enrichment_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--awards',type=Path,required=True);p.add_argument('--legacy',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--events',type=Path);p.add_argument('--bands',type=Path);p.add_argument('--output',type=Path,required=True);build(p.parse_args())
