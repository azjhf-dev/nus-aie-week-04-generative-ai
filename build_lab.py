# /// script
# requires-python = ">=3.11"
# dependencies = ['marimo==0.25.2', 'numpy>=2,<3', 'pandas>=2.2,<3', 'torch>=2.6,<3']
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Week 4 build lab")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import json

    return mo, np, pd


@app.cell
def _(np):
    import html as html_lib

    def chart_wrap(title, svg):
        return '<section style="margin:12px 0;padding:14px;background:#f6f8f6;border-radius:12px"><p style="font:600 15px/1.4 system-ui;margin:0 0 10px;color:#20332e">'+html_lib.escape(title)+'</p>'+svg+'</section>'

    def line_graph(series, labels, title, width=400, height=250):
        colors=['#12675c','#b56420','#7353a6','#3465a4']
        all_values=[float(v) for s in series for v in s]
        low=min(all_values+[0]); high=max(all_values+[0]); span=max(high-low,1e-6)
        n=max(len(s) for s in series); left=48; right=width-15; top=14; bottom=height-48
        pieces=[f'<svg role="img" aria-label="{html_lib.escape(title)}" viewBox="0 0 {width} {height}" style="width:100%;height:auto"><title>{html_lib.escape(title)}</title>']
        for frac in [0,.5,1]:
            y=bottom-frac*(bottom-top); val=low+frac*span
            pieces.append(f'<path d="M{left},{y} H{right}" stroke="#dce5df"/><text x="1" y="{y+4}" font-size="14" fill="#435a50">{val:.2f}</text>')
        for i,(s,label) in enumerate(zip(series,labels)):
            pts=' '.join(f'{left+j/max(n-1,1)*(right-left):.2f},{bottom-(float(v)-low)/span*(bottom-top):.2f}' for j,v in enumerate(s))
            pieces.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i%4]}" stroke-width="2.5"/>')
        pieces.append('</svg>')
        legend='<p style="font:13px/1.7 system-ui;margin:2px 0 0">'+' &nbsp; '.join(f'<span style="color:{colors[i%4]}">&#9679; {html_lib.escape(label)}</span>' for i,label in enumerate(labels))+'</p>'
        return chart_wrap(title,''.join(pieces)+legend)

    def bars(values,labels,title):
        colors=['#12675c','#b56420','#7353a6','#3465a4']; n=len(values); maxv=max([float(v) for v in values]+[1e-8]); h=12+44*n
        s=[f'<svg role="img" aria-label="{html_lib.escape(title)}" viewBox="0 0 400 {h}" style="width:100%;height:auto"><title>{html_lib.escape(title)}</title>']
        for i,(v,label) in enumerate(zip(values,labels)):
            y=4+44*i; w=max(0,float(v)/maxv*210)
            s.append(f'<text x="0" y="{y+18}" font-size="14" fill="#20332e">{html_lib.escape(str(label)[:16])}</text><rect x="132" y="{y}" width="{w:.2f}" height="25" rx="4" fill="{colors[i%4]}"/><text x="{137+w:.2f}" y="{y+18}" font-size="14" fill="#20332e">{float(v):.3f}</text>')
        return chart_wrap(title,''.join(s)+'</svg>')

    def heat(matrix,title):
        a=np.asarray(matrix); rows,cols=a.shape; mn=float(a.min()); mx=float(a.max()); span=max(mx-mn,1e-9)
        cw=min(65,380/max(cols,1)); ch=min(55,300/max(rows,1)); w=cols*cw+14; h=rows*ch+14
        parts=[f'<svg role="img" aria-label="{html_lib.escape(title)}" viewBox="0 0 {w} {h}" style="width:100%;max-width:500px;height:auto"><title>{html_lib.escape(title)}</title>']
        for i in range(rows):
            for j in range(cols):
                v=float(a[i,j]); t=(v-mn)/span; color=f'rgb({int(236-210*t)},{int(244-130*t)},{int(239-135*t)})'
                x=7+j*cw;y=7+i*ch
                parts.append(f'<rect x="{x}" y="{y}" width="{cw-2}" height="{ch-2}" rx="3" fill="{color}"/><text x="{x+cw/2}" y="{y+ch*.62}" text-anchor="middle" fill="{"white" if t>.58 else "#20332e"}" font-size="16">{v:.2f}</text>')
        return chart_wrap(title,''.join(parts)+'</svg>')

    def scatter(real,sample,title):
        arrays=[np.asarray(real),np.asarray(sample)]
        allp=np.concatenate(arrays);low=allp.min(axis=0);high=allp.max(axis=0);span=np.maximum(high-low,1e-6)
        pieces=['<svg role="img" aria-label="Original and generated point clouds" viewBox="0 0 400 300" style="width:100%;height:auto">']
        for a,color in zip(arrays,['#12675c','#b56420']):
            for xy in a:
                x,y=(xy-low)/span;pieces.append(f'<circle cx="{20+x*360:.2f}" cy="{275-y*250:.2f}" r="2.2" fill="{color}" opacity=".5"/>')
        pieces.append('</svg><p style="font:13px system-ui;color:#435a50">Green: training points. Orange: newly generated points. Axes share the same scale within this plot.</p>')
        return chart_wrap(title,''.join(pieces))

    def flow(nodes,enabled=True):
        parts=['<style>@keyframes labPulse{0%,100%{opacity:.35}50%{opacity:1}}@media(prefers-reduced-motion:reduce){.lab-pulse{animation:none!important}}</style>', '<svg role="img" aria-label="Animated conceptual data flow" viewBox="0 0 400 235" style="width:100%;height:auto"><title>Conceptual data flow. Motion shows order, not measured time.</title>']
        for i,name in enumerate(nodes):
            col=i%2;row=i//2;x=8+col*195;y=12+row*98
            sty=f'animation:labPulse 3s {i*.45}s infinite' if enabled else ''
            parts.append(f'<g class="lab-pulse" style="{sty}"><rect x="{x}" y="{y}" width="180" height="70" rx="11" fill="#12675c"/><text x="{x+90}" y="{y+42}" text-anchor="middle" fill="white" font-size="19">{i+1}. {html_lib.escape(name)}</text></g>')
        parts.append('</svg><p style="font:13px/1.4 system-ui;color:#567068">Conceptual sequence, not a measured timeline. Pause with the checkbox above.</p>')
        return ''.join(parts)

    return line_graph, scatter


@app.cell
def _(mo):
    mo.md("""
    # Week 4: build and test

    This second notebook moves from the browser mechanism lab to a working project. **Run it locally from the repository root.** The hosted page is an executed read-only snapshot. The PyTorch and LangGraph builds are not claimed to execute inside a mobile browser.

    No API key, model download or paid compute is required for these default projects. Synthetic tasks are deliberately small. [Official scope](https://www.comp.nus.edu.sg/executive-education/course/sctp-generative-ai-and-its-applications/).
    """)
    return


@app.cell
def _():
    from generative_training import train_denoising, train_vae, train_language

    return train_denoising, train_language, train_vae


@app.cell
def _(mo):
    generator_type=mo.ui.dropdown(["VAE","DDPM"],value="VAE",label="Train small generative model")
    generation_steps=mo.ui.slider(100,1000,step=100,value=400,label="Optimizer steps")
    mo.vstack([generator_type,generation_steps])
    return generation_steps, generator_type


@app.cell
def _(generation_steps, generator_type, train_denoising, train_vae):
    if generator_type.value=='VAE':
        generated_real,generated_sample,generative_loss=train_vae(generation_steps.value)
    else:
        generated_real,generated_sample,generative_loss=train_denoising(generation_steps.value)
    return generated_real, generated_sample, generative_loss


@app.cell
def _(
    generated_real,
    generated_sample,
    generative_loss,
    line_graph,
    mo,
    pd,
    scatter,
):
    mo.vstack([mo.Html(line_graph([generative_loss],['training objective'],'Actual generative model optimization')),mo.Html(scatter(generated_real,generated_sample,'Training distribution and new generated samples')),mo.ui.table(pd.DataFrame(generated_sample[:20],columns=['sample x','sample y']),selection=None),mo.md('Inspect the point cloud in the source task. A decreasing training objective alone does not establish sample quality, coverage or non-memorisation.')])
    return


@app.cell
def _(mo):
    objective=mo.ui.dropdown(["causal","masked"],value="causal",label="Train language objective")
    objective
    return (objective,)


@app.cell
def _(objective, train_language):
    language_result=train_language(objective.value)
    return (language_result,)


@app.cell
def _(language_result, line_graph, mo):
    mo.vstack([mo.Html(line_graph([language_result['loss']],['cross entropy'],'Transformer objective training')),mo.md(f"Held-out synthetic token accuracy **{language_result['held_out_token_accuracy']:.3f}** across {language_result['test_sequences']} sequences. This periodic task is intentionally easy and is not natural-language capability.")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Architecture boundaries

    The VAE learns a posterior and a KL-regularised reconstruction loss. The DDPM learns noise prediction with a forward schedule and performs a reverse stochastic update on **2D points**. The language lab trains an encoder with a causal attention mask or a masked-token objective on a six-symbol grammar.

    These are not pretrained BERT, GPT or Stable Diffusion, and the DDPM is not latent or text-conditioned. No borrowed weights or unlicensed images are bundled.

    ## Build challenge

    Plot generated versus training samples, quantify mode coverage, run multiple seeds, and add a held-out noise-prediction set. For real pretrained model use, review the model card, license, hardware and download size before opting in. Do not buy hosted inference.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## End-of-week evidence

    Save your lab log, test output and a short explanation of one failure. Complete ASSESSMENT.md before opening its solutions. Keep source permissions and version information with every borrowed dataset or model.
    """)
    return


if __name__ == "__main__":
    app.run()
