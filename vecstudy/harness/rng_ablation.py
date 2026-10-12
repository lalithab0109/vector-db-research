"""Paired interpretation only; evaluation uses the shared R2 harness."""
def paired_effects(rows):
    pairs=[]
    for off,on in zip(rows[::2],rows[1::2],strict=True):
        assert not off['rng'] and on['rng']
        for field in ('target_bloat','delta','selected_count','snapshot_sha256','query_count','npost','k'):
            assert off[field]==on[field], f'Pair mismatch: {field}'
        assert on['replicated_count'] <= off['replicated_count']
        assert on['actual_bloat'] <= off['actual_bloat']
        assert on['rng_accepted_count']+on['rng_rejected_count']==off['replicated_count']
        bloat=off['actual_bloat']-on['actual_bloat']
        scans=off['scans_per_query']-on['scans_per_query']
        pairs.append(dict(target_bloat=off['target_bloat'], delta=off['delta'],
                          bloat_reduction=bloat,relative_bloat_reduction=bloat/off['actual_bloat'],
                          recall_difference=on['recall_at_10']-off['recall_at_10'],
                          scans_per_query_difference=-scans,scan_reduction=scans,
                          relative_scan_reduction=scans/off['scans_per_query']))
    return pairs


def plot_rng(rows,baseline,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.scatter(baseline['actual_bloat']*100,baseline['recall_at_10'],marker='s',color='black',label='R5 baseline')
    for i,(off,on) in enumerate(zip(rows[::2],rows[1::2])):
        ax.plot([off['actual_bloat']*100,on['actual_bloat']*100],
                [off['recall_at_10'],on['recall_at_10']],'-',color='gray',alpha=.6)
        for r,marker,color,label in ((off,'o','tab:blue','Without RNG'),(on,'^','tab:orange','With RNG')):
            ax.scatter(r['actual_bloat']*100,r['recall_at_10'],marker=marker,color=color,label=label if i==0 else None)
            ax.annotate(f"{r['target_bloat']*100:g}%"+(' + RNG' if r['rng'] else ''),
                (r['actual_bloat']*100,r['recall_at_10']),xytext=(5,7 if not r['rng'] else -15),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Actual bloat (%)',ylabel='Recall@10',title='Same-delta SPANN-style RNG ablation on SIFT1M')
    ax.legend();ax.grid(alpha=.25);ax.margins(x=.2,y=.2);fig.tight_layout()
    fig.savefig(path,dpi=200);plt.close(fig)
