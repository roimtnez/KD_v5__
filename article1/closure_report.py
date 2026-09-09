"""Scientific questions and closure status derived only from the current snapshot."""
import json
from pathlib import Path
import pandas as pd


def markdown_table(frame):
    def value(v):
        return str(v).replace('|','/').replace('\n',' ')
    lines = ['| '+' | '.join(map(value,frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    return '\n'.join(lines+['| '+' | '.join(map(value,row))+' |' for row in frame.itertuples(index=False,name=None)])


def write_report(out: Path):
    from article1.progress import read_csv, save_table
    manifest = json.loads((out/'manifest.json').read_text())
    main = manifest.get('main_analysis',{})
    supervised = manifest.get('supervised_analysis',{})
    curve = manifest.get('proxy_curve',{})
    masks = read_csv(out/'tables/source_inventory.csv')
    masks_closed = len(masks)==54 and masks.status.eq('present').all()
    primary_closed = main.get('status')=='closed'
    ce_closed = supervised.get('status')=='closed'
    curve_closed = curve.get('complete') and curve.get('mode')=='definitive' and curve.get('complete_pairs')==45
    status = lambda closed: 'cerrado' if closed else 'provisional'
    observations = []
    def add(question, contrast, artifact, observation, interpretation, limitation, state):
        observations.append(dict(zip(['Pregunta científica','Contraste','Figura o tabla','Observación','Interpretación defendible','Limitación','Estado'],
                                     [question,contrast,artifact,observation,interpretation,limitation,state])))
    add('¿Qué acredita M?', 'Expertise por cliente y clase', 'tables/expertise_cells.csv; figures/expertise_*',
        f'{int(masks.status.eq("present").sum())}/54 condiciones verificadas',
        'Acredita accuracy condicional de clase con los umbrales fijados',
        'La evidencia seleccionada no valida independientemente generalización ni rechazo OOD', status(masks_closed))
    specs = [('Selección','selection_logit','ORACLE-logit − FedDF-logit'),
             ('Operador FedDF','feddf_pooling','FedDF-prob − FedDF-logit'),
             ('Operador ORACLE','oracle_pooling','ORACLE-prob − ORACLE-logit'),
             ('Expertise','expertise_gain','EXPERT-prob − FedDF-prob'),
             ('Gap ORACLE','oracle_expertise_gap','ORACLE-prob − EXPERT-prob'),
             ('Soporte','support','EXPERT-prob-SR − EXPERT-prob')]
    for question, name, contrast in specs:
        paired = read_csv(out/'tables'/f'{name}_paired.csv')
        add(question,contrast,f'tables/{name}_paired.csv; figures/{name}_student_test_accuracy.png',
            f'{len(paired)} pares; accuracy y NLL, media y SD por dataset/régimen',
            'Comparación del procedimiento bajo receta y fuentes compartidas',
            'Tres seeds; ORACLE no es cota garantizada; SR no demuestra destrucción de dark knowledge',status(primary_closed))
    add('¿Qué añade el conocimiento privado frente a etiquetas públicas?', 'EXPERT-prob − CE, N=10000',
        'tables/private_knowledge_summary.csv; figures/private_knowledge_*',
        f'{supervised.get("complete_pairs",0)}/54 pares; {supervised.get("present_ce",0)}/9 CE únicos',
        'Cuantifica la diferencia del sistema KD frente a CE con el mismo proxy etiquetado',
        'Solo N=10000; también cambia el objetivo CE/KD; CE reutilizado no crea nuevas réplicas',status(ce_closed))
    add('¿Cómo cambia KD−CE con N?', 'CIFAR, IID/alpha0p1/single; 1200 updates',
        'tables/proxy_budget_completeness.csv; tables/proxy_curve_summary.csv; figures/curve_*',
        f'{curve.get("present",0)}/60 ejecuciones, {curve.get("complete_pairs",0)}/45 pares',
        'Se describen únicamente tamaños y seeds observados',
        'Huecos no son ceros; seeds distintas no definen medias comparables; cruce no es óptimo',status(curve_closed))
    add('¿Aportan otros controles?', 'EXPERT-logit, Confidence, Consensus, Energy',
        'tables/contrast_summary.csv; tables/*_excluded.csv',
        'Se conservan ejecuciones adicionales verificadas del respaldo, sin contarlas como baseline nuevo',
        'Contrastes adicionales solo en condiciones emparejadas',
        'Su incompletitud no impide cerrar el diseño principal de seis métodos','provisional')
    questions = pd.DataFrame(observations)
    save_table(out, 'scientific_questions', questions)
    manifest['closure'] = dict(masks=status(masks_closed), baseline=status(primary_closed),
                               supervised=status(ce_closed), proxy_curve=status(curve_closed))
    manifest.setdefault('versions', {}).update(pandas=pd.__version__, protocol='article1-v3')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    text = '# Artículo 1: cierre principal y curva '+('completa' if curve_closed else 'provisional')+'\n\n'
    text += f"Instantánea UTC: {manifest['snapshot_date']}. Análisis: `{manifest['analysis_commit']}`. "
    text += f"Checkout de entrenamiento observado: `{manifest['active_checkout_commit']}`. Diseño principal detectado: **{manifest['baseline_design']}**.\n\n"
    text += 'La configuración actual no se atribuye retroactivamente a procesos o caches. Las rutas, SHA256, fechas, filas y procedencia declarada están en manifest.json; los CSV se leen exclusivamente de la instantánea.\n\n'
    text += markdown_table(questions)+'\n\n'
    summary = read_csv(out/'tables/private_knowledge_summary.csv')
    if not summary.empty:
        acc = summary[summary.metric.eq('delta_student_test_accuracy')].copy()
        nll = summary[summary.metric.eq('delta_student_test_nll')]
        table = acc[['dataset','regime','mean','sd','n','seeds']].merge(nll[['dataset','regime','mean','sd']],on=['dataset','regime'],suffixes=('_accuracy','_nll'))
        table['accuracy_pp'] = (100*table.mean_accuracy).round(3)
        table['SD_pp'] = (100*table.sd_accuracy).round(3)
        table = table[['dataset','regime','accuracy_pp','SD_pp','mean_nll','sd_nll','n','seeds']].round(4)
        text += '## Conocimiento privado frente a CE, N=10000\n\n'
        text += markdown_table(table)+'\n\n'
        for dataset, rows in table.groupby('dataset'):
            better = rows[rows.accuracy_pp>0].regime.tolist()
            worse = rows[rows.accuracy_pp<0].regime.tolist()
            text += f"{dataset}: diferencia media de accuracy positiva en {better or 'ningún régimen'} y negativa en {worse or 'ningún régimen'}. "
            text += 'Es una descripción de estas tres seeds, no una declaración de superioridad universal o equivalencia.\n\n'
    curve_summary = read_csv(out/'tables/proxy_curve_summary.csv')
    if not curve_summary.empty:
        acc = curve_summary[curve_summary.metric.eq('delta_student_test_accuracy')]
        nll = curve_summary[curve_summary.metric.eq('delta_student_test_nll')]
        table = acc[['regime','proxy_size','mean','sd','n','seeds']].merge(
            nll[['regime','proxy_size','mean','sd']], on=['regime','proxy_size'], suffixes=('_accuracy','_nll'))
        table['accuracy_pp'] = (100*table.mean_accuracy).round(3)
        table['SD_pp'] = (100*table.sd_accuracy).round(3)
        text += '## Curva CIFAR: EXPERT-prob − CE\n\n'
        text += markdown_table(table[['regime','proxy_size','accuracy_pp','SD_pp','mean_nll','sd_nll','n','seeds']].round(4))+'\n\n'
        for regime, rows in table.groupby('regime'):
            rows = rows.sort_values('proxy_size')
            intervals = []
            for left, right in zip(rows.iloc[:-1].itertuples(), rows.iloc[1:].itertuples()):
                if left.seeds == right.seeds and left.mean_accuracy*right.mean_accuracy < 0:
                    intervals.append(f'{left.proxy_size}–{right.proxy_size}')
            text += f'{regime}: cambios de signo de la diferencia media de accuracy entre tamaños adyacentes con las mismas seeds: {", ".join(intervals) or "ninguno observado"}. '
            text += 'Son intervalos descriptivos de estas seeds, sin demostrar un mínimo suficiente ni un tamaño óptimo.\n\n'
    text += '## Interpretación y límites\n\n'
    text += 'Las figuras principales muestran puntos por seed, diferencias emparejadas, media y SD muestral. No se agrupan los tres datasets como réplicas intercambiables. '
    text += 'Las relaciones entre cobertura, soporte y efecto de expertise están en coverage_support_results.csv y figuras descriptivas sin ajuste causal. Dirichlet también altera cantidades; los regímenes no aíslan exclusivamente especialización.\n\n'
    text += 'Validation selecciona checkpoints; expertise construye M. No hay test local independiente en v3 ni curvas por época almacenadas. Los soportes escasos no introducen nuevas reglas. '
    text += 'No se concluye equivalencia por falta de significación, ni pérdida causal de dark knowledge por un efecto de SR. No se eligieron temperaturas, thresholds o variantes con test.\n\n'
    text += manifest['checkpoint_limitation']+'\n\n'
    text += 'No se entrenó, no se infirió de nuevo, no se usó CUDA y no se modificó el checkout activo. Los CSV opcionales vacíos no acreditan ejecuciones; los duplicados de maestro/bloques se reutilizan solo tras comprobar igualdad.\n\n'
    text += ('## Procedimiento final completado; futuras actualizaciones\n\n' if curve_closed else '## Cuando termine proxy-curve\n\n')
    text += '1. Tomar una nueva instantánea y auditar las 60 ejecuciones CIFAR y los 45 pares.\n2. Ejecutar el notebook de presupuesto con modo definitivo y regenerar figuras de N.\n3. Revisar conclusiones y captions con las seeds realmente disponibles.\n4. Separar las preguntas resueltas de las pendientes.\n5. Decidir posteriormente si estudiar temperatura, más controles o viabilidad unlabeled; aquí no se implementa ni lanza unlabeled.\n\n'
    text += 'Un cruce solo se localiza entre N evaluados, no demuestra un tamaño óptimo. El presupuesto iguala actualizaciones, no necesariamente ejemplos consumidos; examples_consumed.csv conserva ambos.\n'
    (out/'report.md').write_text(text)
    return text
