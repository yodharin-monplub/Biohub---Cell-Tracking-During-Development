# -*- coding: utf-8 -*-
"""Generator code of the public CC0 "Biohub Synthetic Dataset", extracted for LOCAL generation.

PROVENANCE
  Author   : Jose Freitas (Kaggle user josefreitasalvesneto)
  Notebook : josefreitasalvesneto/biohub-synthetic-dataset  (biohub-synthetic-dataset.ipynb)
  Local copy used: C:\\biohub_data\\public\\josefreitasalvesneto__biohub-synthetic-dataset\\biohub-synthetic-dataset.ipynb
  Licence  : CC0 (stated by the author for the dataset / its explorer notebook)

This file is ASSEMBLED MECHANICALLY by copying line ranges of the notebook's code cells (see the
"VERBATIM" banners; cell numbers are indices into nb["cells"], markdown cells included). Do not edit the
verbatim blocks by hand. Only the blocks marked "ADAPTED" differ from the notebook:
  * find_train_dir(): Kaggle /kaggle/input paths -> local Data\\competition\\train (or $BIOHUB_TRAIN_DIR).
  * generate_sequence(): the BODY of the `while` loop of run_dataset_build() part B (cell 15), moved
    into a function of the sequence index j so sequences can be generated in parallel / resumed.
    Physics, sampling order and RNG seeds (500000+j for motion, 900000+j*97+t for rendering) are
    unchanged, so with seed=0 and the same calibration movies sequence j is the notebook's seq_j.
  * calibrate(): the first lines of run_dataset_build() (choice of the 10 calibration movies + _real_stats).
Nothing of the notebook's other generators (template / parametric / liquid model, validation, figures,
static volumes) is needed for the time sequences and none of it is included.

Effective notebook configuration for the published build (cell 0): DS_SEQ_LEN=6, DS_DIV_RATE=0.05,
SYNTH_PLACE_MODE=shell, SYNTH_EDGE_FRAC=0.35; every other SYNTH_* knob read below is left at the default
written in the code (no cell that runs in that configuration mutates os.environ).
"""
import os, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter



# ================================================================================================
# VERBATIM cell 2 (constants used by the DoG detector / pooling)
# ================================================================================================

VOXEL_UM = np.array([1.625, 0.40625, 0.40625]); DOWNSAMPLE = (1, 4, 4)
EFF_VOXEL = VOXEL_UM * np.array(DOWNSAMPLE)


# ================================================================================================
# VERBATIM cell 4, lines 1-14 (pool_xy, DoG detector)
# ================================================================================================

def pool_xy(f): return f[::DOWNSAMPLE[0], ::DOWNSAMPLE[1], ::DOWNSAMPLE[2]].astype(np.float32)

def dog_response(vol, scales=((1.5, 4.0), (2.2, 5.5))):
    dog = None
    for ss, sl in scales:
        r = gaussian_filter(vol, tuple(ss / EFF_VOXEL)) - gaussian_filter(vol, tuple(sl / EFF_VOXEL))
        dog = r if dog is None else np.maximum(dog, r)
    return dog

def dog_detect_precise(vol, rel_thr=0.35):
    dog = dog_response(vol)
    size = tuple(np.maximum(1, (2 * np.round(3.0 / EFF_VOXEL) + 1)).astype(int))
    mx = maximum_filter(dog, size=size, mode="nearest")
    return np.argwhere((dog == mx) & (dog >= dog.max() * rel_thr)), dog


# ================================================================================================
# ADAPTED cell 6 find_train_dir (paths only)
# ================================================================================================

def find_train_dir():
    env = os.environ.get("BIOHUB_TRAIN_DIR")
    cands = ([Path(env)] if env else []) + [Path(__file__).resolve().parents[2] / "Data" / "competition" / "train"]
    for b in cands:
        if b.exists() and any(b.glob("*.zarr")): return b
    return None



# ================================================================================================
# VERBATIM cell 6, lines 9-22 (read_frame, norm)
# ================================================================================================

def read_frame(zp, t):
    meta = json.loads((zp/"0"/"zarr.json").read_text()); shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"]); fs = shape[1:]
    try:
        import blosc2
        a = np.frombuffer(blosc2.decompress((zp/"0"/"c"/str(t)/"0"/"0"/"0").read_bytes()), dtype=dtype)
        if a.size == int(np.prod(fs)): return a.reshape(fs).copy(), shape
    except Exception: pass
    import zarr
    return np.asarray(zarr.open(zp/"0", mode="r")[t]), shape

def norm(f):
    lo, hi = np.percentile(f, 1), np.percentile(f, 99.5)
    return np.clip((f - lo)/max(hi - lo, 1e-6), 0, 1).astype(np.float32)


# ================================================================================================
# VERBATIM cell 13, lines 5-332 (native generator: sphere, shell fit/sampling, gen_volume_native)
# ================================================================================================

# voxel fisico NATIVO (antes do pool)
VOXEL_NATIVE = np.array([1.625, 0.40625, 0.40625])
POOL = 4                                                # XY downsample do pipeline (DOWNSAMPLE=(1,4,4))


def _sphere_native(r_um, rng, sharp=7, ellip=0.5, frac=None):
    """Esfera FISICA de raio r_um, amostrada na grade NATIVA anisotropica (achatada em z). Perfil
    super-gaussiano (disco preenchido, borda nitida) + variacao eliptica.
    `frac` = deslocamento SUB-VOXEL (p - round(p)). Sem ele a celula so' pode ficar centrada num voxel
    INTEIRO e o rotulo de centroide sai quantizado (erro sistematico de ate 0.5 voxel nativo, que o
    detector aprende como ruido no alvo). Com ele o perfil e' amostrado deslocado e o centroide
    verdadeiro (float) e' o rotulo."""
    axes_vox = r_um / VOXEL_NATIVE                       # semi-eixos em VOXELS nativos (z pequeno, xy grande)
    aniso = rng.uniform(0.82, 1.20, 3)
    if rng.random() < ellip: aniso[rng.integers(3)] *= rng.uniform(1.4, 2.0)   # eliptica marcada
    axes = np.maximum(axes_vox * aniso, 0.6)
    size = (np.ceil(axes) * 2 + 3).astype(int)          # bounding box por eixo
    zz, yy, xx = np.indices(tuple(size)); c = size // 2
    fz, fy, fx = (0.0, 0.0, 0.0) if frac is None else (float(frac[0]), float(frac[1]), float(frac[2]))
    # rotacao so' no plano XY (a anisotropia z e' fisica, nao rotaciona junto)
    th = rng.uniform(0, np.pi); ct, stt = np.cos(th), np.sin(th)
    dy = (yy - c[1] - fy); dx = (xx - c[2] - fx); dz = (zz - c[0] - fz)
    ry = ct*dy - stt*dx; rx = stt*dy + ct*dx
    rn = np.sqrt((dz/axes[0])**2 + (ry/axes[1])**2 + (rx/axes[2])**2)
    return np.exp(-(rn**sharp)).astype(np.float32)


# ============================================================================================
# CASCA DO EMBRIAO (algoritmo NOVO) -- colocacao GERATIVA com a estrutura espacial do real.
#
# PROBLEMA que ele resolve: o processo de Thomas (clusters aleatorios) espalha grumos pelo volume
# inteiro; o real tem uma FAIXA CURVA de tecido com VAZIO ESCURO de um lado (a blastoderme sobre o
# vitelo atravessando o campo). Copiar as posicoes reais (`cens_native`) casa a distribuicao por
# construcao mas NAO e' gerativo -- reusa os poucos frames reais.
# SOLUCAO: ajustar a GEOMETRIA (nao as posicoes) da casca a partir do real -- centro/raio de uma
# esfera + espessura -- e depois AMOSTRAR cascas novas. Sai estrutura coerente e inedita.
#
# Ajuste algebrico de esfera (Pratt/Coope): p/ pontos p_i, |p_i|^2 = 2 p_i . c + (R^2 - |c|^2),
# linear em [c, k] com k = R^2-|c|^2 -> minimos quadrados. R = sqrt(k + |c|^2).
# LIMITE PLANO: embriao grande => R >> campo de visao e a esfera vira um PLANO; o ajuste fica mal
# condicionado, entao caimos p/ PCA (normal = autovetor de menor variancia) -- e' o caso R->inf.
# ============================================================================================

def fit_shell(pts_vox, voxel=None):
    """Ajusta a casca (esfera OU plano) a pontos em voxels NATIVOS. Retorna dict com geometria em um.
    `pts_vox` = deteccoes/centroides reais (N,3) em (z,y,x) voxel nativo."""
    voxel = VOXEL_NATIVE if voxel is None else np.asarray(voxel, float)
    P = np.atleast_2d(np.asarray(pts_vox, float)) * voxel                    # -> um (isotropico)
    n = len(P)
    if n < 12: return dict(kind="none", n=n)
    c0 = P.mean(0); Q = P - c0
    # --- tentativa ESFERA ---
    A = np.concatenate([2.0*Q, np.ones((n, 1))], 1)
    b = (Q**2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cc = sol[:3]; k = sol[3]
        R2 = k + float(cc @ cc)
        R = float(np.sqrt(R2)) if R2 > 0 else -1.0
        center = c0 + cc
        rad = np.linalg.norm(P - center, axis=1)
        resid = float(np.std(rad - R)) if R > 0 else np.inf
    except Exception:
        R, resid, center = -1.0, np.inf, c0
    # --- alternativa PLANO (limite R->inf) via PCA ---
    U, S, Vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
    normal = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)
    off = (P - c0) @ normal
    resid_plane = float(np.std(off))
    # extensao lateral do tecido (p/ saber se ele cobre todo o campo ou so' uma faixa)
    span = np.percentile(P, [2, 98], axis=0)
    # ESCOLHA: esfera so' vence se for bem condicionada (R finito, nao absurdo) E ajustar melhor
    fov = float(np.linalg.norm(P.max(0) - P.min(0)) + 1e-9)
    # PISO FISICO DE RAIO (defeito encontrado NA FIGURA, invisivel nas metricas KS): o ajuste devolvia
    # R=50um num campo de 104um -- esfera MENOR que o campo de visao, que cabe inteira dentro do volume.
    # Celulas numa casca dessas formam uma BOLA OCA e a projecao em Z sai como ANEL com centro escuro.
    # O real NUNCA tem vazio central: a borda dele e' quase reta, ou seja R >> campo. O ajuste estava
    # SUPERAJUSTANDO curvatura local de poucas deteccoes. Agora a esfera so' e' aceita se for
    # fisicamente plausivel (raio >= 1.5x o campo); abaixo disso usamos o PLANO (limite R->inf).
    R_MIN = float(os.environ.get("SYNTH_SHELL_RMIN_FOV", "1.5")) * fov
    sphere_ok = (R > R_MIN) and (R < 60.0 * fov) and np.isfinite(resid) and (resid < 0.92 * resid_plane)
    if sphere_ok:
        return dict(kind="sphere", center_um=center, R_um=R, thick_um=max(resid, 1.0),
                    n=n, resid=resid, resid_plane=resid_plane, span_um=span)
    return dict(kind="plane", point_um=c0, normal=normal, thick_um=max(resid_plane, 1.0),
                n=n, resid=float(resid) if np.isfinite(resid) else -1.0, resid_plane=resid_plane, span_um=span)


def _density_field(rng, shape_zyx, smooth_vox=14.0, contrast=1.0):
    """Campo de densidade suave em [0,1]: regioes mais/menos povoadas ao longo do tecido (o real nao
    e' uniforme dentro da faixa). Ruido branco borrado = campo de baixa frequencia."""
    f = rng.normal(0, 1, shape_zyx).astype(np.float32)
    f = gaussian_filter(f, smooth_vox)
    f -= f.min(); f /= (f.max() + 1e-9)
    return np.clip(0.5 + contrast * (f - 0.5), 0.0, 1.0)


def sample_shell_positions(n_cells, shape_zyx, rng, shell=None, r_lo=2.5, r_hi=4.5, csf=1.0,
                           jitter=0.25, dens_contrast=1.0):
    """Coloca `n_cells` numa CASCA (esfera/plano) atravessando o volume, com gradiente de densidade e
    checagem de colisao. `shell` = saida de fit_shell (geometria medida no real); None -> sorteia uma.
    Retorna (pos_voxel (N,3), radii_um (N,))."""
    Z, Y, X = shape_zyx
    lo = np.array([4.0, 12.0, 12.0]); hi = np.array([Z-4.0, Y-12.0, X-12.0])
    vox = VOXEL_NATIVE
    ext_um = (np.array([Z, Y, X]) * vox)                                     # campo em um
    # ---- geometria: usa a medida OU sorteia uma plausivel (variacao entre embrioes/frames) ----
    if shell is None or shell.get("kind") not in ("sphere", "plane"):
        kind = "sphere" if rng.random() < 0.65 else "plane"
        R = float(rng.uniform(1.2, 4.0) * float(ext_um.max()))               # curvatura suave
        nrm = rng.normal(0, 1, 3); nrm /= np.linalg.norm(nrm) + 1e-9
        cen = ext_um * 0.5 + nrm * R * float(rng.uniform(0.75, 1.05))        # centro FORA do campo
        shell = dict(kind=kind, center_um=cen, R_um=R, point_um=ext_um*rng.uniform(0.35, 0.65, 3),
                     normal=nrm, thick_um=float(rng.uniform(6.0, 16.0)))
    else:
        shell = dict(shell)
        # PERTURBA a geometria medida -> casca NOVA (gerativo), nao a mesma do frame real
        j = float(jitter)
        if shell["kind"] == "sphere":
            shell["R_um"] = float(shell["R_um"] * rng.normal(1.0, 0.10*j/0.25 if j else 0.0)) if j else shell["R_um"]
            shell["center_um"] = np.asarray(shell["center_um"], float) + rng.normal(0, j*12.0, 3)
        else:
            nrm = np.asarray(shell["normal"], float) + rng.normal(0, j*0.18, 3)
            shell["normal"] = nrm / (np.linalg.norm(nrm) + 1e-9)
            shell["point_um"] = np.asarray(shell["point_um"], float) + rng.normal(0, j*10.0, 3)
        # REGIME DE BORDA (defeito visto na figura V01): o real tem DOIS regimes -- campos DENTRO do
        # tecido, que preenchem o quadro, e campos na BORDA do embriao, com uma faixa de celulas e um
        # grande vazio escuro do outro lado. Nossa casca vinha da calibracao (videos de interior), com
        # a normal ao longo de Z -> a laje fica paralela a imagem e SEMPRE cobre tudo na projecao.
        # Para sair a FAIXA, a normal precisa estar NO PLANO da imagem (laje atravessando de lado).
        # Aqui uma fracao dos volumes recebe essa reorientacao, reproduzindo o regime de borda.
        if rng.random() < float(os.environ.get("SYNTH_EDGE_FRAC", "0.35")):
            ang = rng.uniform(0, 2*np.pi)
            n_edge = np.array([rng.uniform(-0.25, 0.25), np.cos(ang), np.sin(ang)], float)
            shell = dict(shell, kind="plane", normal=n_edge/(np.linalg.norm(n_edge)+1e-9),
                         point_um=np.array([Z, Y, X], float)*VOXEL_NATIVE*rng.uniform(0.25, 0.75, 3),
                         thick_um=float(rng.uniform(14.0, 34.0)))
        shell["thick_um"] = float(max(3.0, shell["thick_um"] * rng.normal(1.0, 0.15)))
    thick = float(shell["thick_um"])
    dens = _density_field(rng, (Z, Y, X), smooth_vox=float(os.environ.get("SYNTH_SHELL_DENS_SMOOTH", "14")),
                          contrast=dens_contrast)
    # ---- MAPA DE PROBABILIDADE vetorizado (substitui rejeicao pura, que era fragil e lenta):
    # peso(voxel) = gaussiana(distancia a' superficie) * densidade. Amostrar deste mapa GARANTE que as
    # celulas caiam na casca e cobre o volume todo, sem estourar tentativas.
    gz, gy, gx = np.meshgrid(np.arange(Z, dtype=np.float32), np.arange(Y, dtype=np.float32),
                             np.arange(X, dtype=np.float32), indexing="ij")
    QZ, QY, QX = gz*vox[0], gy*vox[1], gx*vox[2]

    def _dist_map(sh):
        if sh["kind"] == "sphere":
            c = np.asarray(sh["center_um"], float)
            return np.abs(np.sqrt((QZ-c[0])**2 + (QY-c[1])**2 + (QX-c[2])**2) - float(sh["R_um"]))
        p0 = np.asarray(sh["point_um"], float); nn = np.asarray(sh["normal"], float)
        return np.abs((QZ-p0[0])*nn[0] + (QY-p0[1])*nn[1] + (QX-p0[2])*nn[2])

    D = _dist_map(shell)
    # RE-ANCORAGEM: se a casca nao cruza o volume (tudo longe da superficie), ela produziria ZERO
    # celulas. Desloca a geometria p/ que a superficie passe pelo volume, preservando a CURVATURA.
    if float(D.min()) > 1.5 * thick:
        ctr_um = np.array([Z, Y, X], float) * vox * 0.5
        if shell["kind"] == "sphere":
            c = np.asarray(shell["center_um"], float)
            u = (ctr_um - c); nu = np.linalg.norm(u) + 1e-9
            shell["center_um"] = ctr_um - (u/nu) * float(shell["R_um"])   # superficie passa no centro
        else:
            shell["point_um"] = ctr_um
        D = _dist_map(shell)
    W = np.exp(-0.5 * (D / (0.5*thick + 1e-9))**2) * dens
    # zera bordas (celula precisa caber inteira no volume)
    W[: int(lo[0]), :, :] = 0; W[int(hi[0]):, :, :] = 0
    W[:, : int(lo[1]), :] = 0; W[:, int(hi[1]):, :] = 0
    W[:, :, : int(lo[2])] = 0; W[:, :, int(hi[2]):] = 0
    tot = float(W.sum())
    if tot <= 0:                                                          # degenerado -> uniforme
        W = np.zeros_like(W); W[int(lo[0]):int(hi[0]), int(lo[1]):int(hi[1]), int(lo[2]):int(hi[2])] = 1.0
        tot = float(W.sum())
    p = (W.ravel() / tot).astype(np.float64)
    # OVERSAMPLE: sorteia mais candidatos do que o necessario, pois a colisao vai descartar parte
    ncand = int(min(len(p), max(8 * n_cells, 4000)))
    idx = rng.choice(len(p), size=ncand, replace=True, p=p)
    cz, cyx = np.divmod(idx, Y * X); cy, cx = np.divmod(cyx, X)
    cand = np.stack([cz, cy, cx], 1).astype(np.float32)
    cand += rng.uniform(-0.5, 0.5, cand.shape)                            # jitter sub-voxel
    cand = np.clip(cand, lo, hi)
    # aceitacao gulosa com checagem de colisao (mesma regra do resto do gerador)
    cand_r = rng.uniform(r_lo, r_hi, len(cand)).astype(np.float32)
    pos, radii = [], []
    for p_i, r in zip(cand, cand_r):
        if len(pos) >= n_cells: break
        if pos:
            dd = np.linalg.norm((np.array(pos) - p_i) * vox, axis=1)
            if (dd < csf * (r + np.array(radii))).any(): continue
        pos.append(p_i); radii.append(float(r))
    return np.array(pos, np.float32).reshape(-1, 3), np.array(radii, np.float32)


def gen_volume_native(zshape=64, xyshape=256, n_cells=200, rng=None, cens_native=None,
                      medium=None, min_sep_um=None, shell=None):
    """Gera um volume NATIVO (zshape, xyshape, xyshape): fundo escuro + esferas fisicas + PSF + ruido.
    Retorna (vol_native, vol_pooled, centroides_native). cens_native (opcional): posicoes reais."""
    if rng is None: rng = np.random.default_rng(0)
    medium = float(os.environ.get("SYNTH_MEDIUM_LEVEL", "0.04")) if medium is None else medium
    # RAIO calibrado por DETECTABILIDADE (nao a olho): com 2.5-4.5 (diam 7um) o DoG achava so' 0.811
    # das celulas COLOCADAS, enquanto no REAL ele acha 0.91-0.94 -> o sintetico era ~20% mais DIFICIL
    # que a realidade (ensina o detector a esperar celulas menores do que existem). Com 3.5-5.5
    # (diam 9um) o recall vai a 0.913 = casa o real, E bate a EDA (diametro real ~9.5um). Raios maiores
    # (diam 10-11um) dao recall 0.94-0.96, o que ULTRAPASSA o real -- sintetico mais facil que a
    # realidade e' pior p/ treino. A separacao que o user queria e' preservada: NN 11.6um vs real ~10-11.
    r_lo = float(os.environ.get("SYNTH_CELL_RUM_LO", "3.5"))     # raio (um)
    r_hi = float(os.environ.get("SYNTH_CELL_RUM_HI", "5.5"))
    sharp = float(os.environ.get("SYNTH_CELL_SHARPNESS", "4.5"))   # calibrado: brilho relativo pico/media do real
    psf_xy = float(os.environ.get("SYNTH_PSF_XY_UM", "0.55"))    # PSF lateral (um) ~ confocal/light-sheet
    psf_z_ratio = float(os.environ.get("SYNTH_PSFZ", "2.5"))     # PSF axial maior (anisotropia optica real)
    # SEPARACAO CONSCIENTE DE COLISAO: duas celulas so' sao aceitas se a distancia centro-a-centro
    # (um) >= FATOR * (raio_i + raio_j). Assim elas ENCOSTAM mas nao se INTERPENETRAM. Antes o min_sep
    # era FIXO em 7um -> celulas de ~10um de diametro se sobrepunham ~3um, e ao aumentar o tamanho a
    # sobreposicao explodia (celulas se batiam, o DoG as fundia -> vizinho-mais-proximo estragado).
    # fator 1.0 = tangentes (sem overlap); <1 = leve encoste (tecido denso real); >1 = folga.
    csf = float(os.environ.get("SYNTH_COLLISION_FACTOR", "1.0"))

    Z, Y, X = zshape, xyshape, xyshape
    vol = np.clip(rng.normal(medium, medium*0.25, (Z, Y, X)), 0, 1).astype(np.float32)  # LIQUIDO escuro

    # posicoes + RAIOS das celulas. cens reais OU colocacao (AGRUPADA ou uniforme) consciente de tamanho.
    # CLUSTERING (SYNTH_CLUSTER=1, default): o embriao real NAO tem celulas aleatorias uniformes -- elas
    # formam GRUPOS/aglomerados (regioes densas e vazias). Processo de Thomas: sorteia CENTROS de cluster
    # e coloca a maioria das celulas em torno deles (gaussiana), + uma fracao uniforme de fundo. Isso da'
    # a estrutura espacial (tipos de agrupamento) que a colocacao aleatoria uniforme nunca reproduz.
    place_mode = os.environ.get("SYNTH_PLACE_MODE", "cluster")     # cluster (Thomas) | shell (casca) | uniform
    if cens_native is not None and len(cens_native) >= 5:
        pos = np.atleast_2d(cens_native).astype(np.float32)
        radii = rng.uniform(r_lo, r_hi, len(pos)).astype(np.float32)
    elif place_mode == "shell":
        # CASCA: geometria (curvatura/espessura) medida no real, perturbada -> layout coerente e INEDITO
        pos, radii = sample_shell_positions(
            n_cells, (Z, Y, X), rng, shell=shell, r_lo=r_lo, r_hi=r_hi, csf=csf,
            jitter=float(os.environ.get("SYNTH_SHELL_JITTER", "0.25")),
            dens_contrast=float(os.environ.get("SYNTH_SHELL_DENS", "1.0")))
    else:
        clustered = os.environ.get("SYNTH_CLUSTER", "1") == "1"
        ncl = max(3, int(n_cells * float(os.environ.get("SYNTH_CLUSTER_FRAC_N", "0.10"))))   # nÂº de aglomerados
        cfrac = float(os.environ.get("SYNTH_CLUSTER_FRAC", "0.72"))               # fracao de celulas em cluster
        spread_um = float(os.environ.get("SYNTH_CLUSTER_SPREAD_UM", "14.0"))      # raio do aglomerado (um)
        centers = rng.uniform([6, 16, 16], [Z-6, Y-16, X-16], size=(ncl, 3)).astype(np.float32) if clustered else None
        sp_vox = spread_um / VOXEL_NATIVE                                         # spread por eixo (voxels)
        pos, radii = [], []; tries = 0
        while len(pos) < n_cells and tries < n_cells * 60:
            tries += 1
            if clustered and rng.random() < cfrac:                               # celula DENTRO de um aglomerado
                cc = centers[rng.integers(ncl)]
                p = cc + rng.normal(0, sp_vox, 3).astype(np.float32)
                p = np.clip(p, [4, 12, 12], [Z-4, Y-12, X-12])
            else:                                                                # celula de FUNDO (uniforme)
                p = rng.uniform([4, 12, 12], [Z-4, Y-12, X-12])
            r = float(rng.uniform(r_lo, r_hi))
            if pos:
                d = np.linalg.norm((np.array(pos)-p) * VOXEL_NATIVE, axis=1)      # um, centro-a-centro
                thr = csf * (r + np.array(radii))
                # `min_sep_um` era um parametro MORTO (aceito na assinatura e nunca usado -> quem
                # passava recebia no-op silencioso). Agora e' um PISO absoluto de separacao (um).
                if min_sep_um: thr = np.maximum(thr, float(min_sep_um))
                if (d < thr).any(): continue                                      # colide -> rejeita
            pos.append(p); radii.append(r)
        pos = np.array(pos, np.float32); radii = np.array(radii, np.float32)

    # EMISSAO/ESPALHAMENTO: cada nucleo emite luz que se espalha no meio (cauda da PSF + scattering no
    # liquido) -> halo dim que preenche o espaco entre celulas com o brilho ROXO do real. Modelado como
    # uma 2a camada: o CENTRO da celula deposita uma fonte pontual que depois e' borrada LARGO e somada
    # (acumula onde ha muitas celulas = glow; escuro onde nao ha). O nucleo NITIDO vai por maximo.
    halo_amp = float(os.environ.get("SYNTH_HALO_AMP", "0.18"))       # BAIXO de proposito: e' o halo (nao o brilho)
    #                                                                  que preenchia as folgas e fundia as celulas
    halo_um = float(os.environ.get("SYNTH_HALO_UM", "2.6"))   # 3.5 deixava o piso longe da
    #                                                     celula em 0.24 vs 0.16 do real          # alcance do espalhamento (um)
    emis = np.zeros((Z, Y, X), np.float32)                          # camada de emissao (soma)
    cens = []
    # BRILHO por celula: VARIADO e (na maioria) ABAIXO da saturacao. Antes era uniform(0.72,1.0) e apos
    # PSF+emissao TODAS batiam em 1.0 -> distribuicao de pico degenerada (pico unico em 1.0), diferente
    # do real que tem brilho variado (cauda longa). Log-normal-ish: media modesta, cauda p/ cima.
    # BRILHO: o REAL SATURA MUITO (~41% das celulas com pico ~1.0, medido no histograma da galeria).
    # O erro NAO era brilho alto -- era a SEPARACAO. O aspecto "massa branca fundida" vinha do HALO
    # preenchendo as folgas entre celulas, nao do brilho. Baixar o brilho p/ 0.35-0.78 deixou o synth
    # APAGADO (2.9% de saturacao vs 41% do real). Config certa = brilho ALTO (casa a saturacao real)
    # + halo BAIXO (mantem as folgas escuras, celulas distintas): 26.5% sat, recall 0.905.
    b_lo = float(os.environ.get("SYNTH_CELL_BRIGHT_LO", "0.42"))
    b_hi = float(os.environ.get("SYNTH_CELL_BRIGHT_HI", "0.92"))
    for p, r_um in zip(pos, radii):                                 # raio JA sorteado na colocacao (consistente)
        bright = float(np.clip(rng.normal((b_lo+b_hi)/2, (b_hi-b_lo)/3), b_lo, b_hi))   # variacao real (nao satura tudo)
        pr = np.round(p); frac = np.asarray(p, float) - pr                              # deslocamento SUB-VOXEL
        esf = _sphere_native(float(r_um), rng, sharp=sharp, frac=frac) * bright         # nucleo NITIDO (sub-voxel)
        rz, ry, rx = np.array(esf.shape) // 2
        z, y, x = pr.astype(int)
        z0, z1 = max(0, z-rz), min(Z, z+rz+1); y0, y1 = max(0, y-ry), min(Y, y+ry+1); x0, x1 = max(0, x-rx), min(X, x+rx+1)
        if z1 <= z0 or y1 <= y0 or x1 <= x0: continue
        es = esf[z0-(z-rz):z0-(z-rz)+(z1-z0), y0-(y-ry):y0-(y-ry)+(y1-y0), x0-(x-rx):x0-(x-rx)+(x1-x0)]
        vol[z0:z1, y0:y1, x0:x1] = np.maximum(vol[z0:z1, y0:y1, x0:x1], es)     # nucleo = fonte pontual
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            # deposita a luz emitida no centro. Usa o BRILHO da celula (nao `es.max()`, que vem do patch
            # ja CORTADO na borda -> celulas de borda emitiam menos que identicas no interior).
            emis[z, y, x] += bright * halo_amp
        cens.append([float(p[0]), float(p[1]), float(p[2])])        # centroide VERDADEIRO (float), nao arredondado
    # espalha a emissao LARGO (anisotropico) e soma ao volume -> glow roxo entre celulas
    sz2 = halo_um * psf_z_ratio / VOXEL_NATIVE[0]; sy2 = halo_um / VOXEL_NATIVE[1]; sx2 = halo_um / VOXEL_NATIVE[2]
    emis = gaussian_filter(emis, (sz2, sy2, sx2))
    # Normaliza pela RESPOSTA DE UMA FONTE (constante do kernel), nao pelo maximo global. Com o max
    # global, o glow de cada celula passava a depender do aglomerado mais brilhante do volume -> dois
    # volumes com densidades diferentes ganhavam brilho por celula diferente. Assim o halo ACUMULA com
    # a densidade (denso brilha, esparso fica escuro), que e' o comportamento fisico desejado.
    kern_peak = 1.0 / ((2.0*np.pi)**1.5 * sz2 * sy2 * sx2 + 1e-12)
    emis = np.clip(emis / max(kern_peak, 1e-12), 0.0, 1.0)
    vol = vol + emis                                                             # ADICIONA a luz emitida (acumula)

    # PSF OPTICA anisotropica na grade nativa (sigma em voxels = sigma_um / voxel_um)
    sz = psf_xy * psf_z_ratio / VOXEL_NATIVE[0]; sy = psf_xy / VOXEL_NATIVE[1]; sx = psf_xy / VOXEL_NATIVE[2]
    vol = np.clip(gaussian_filter(vol, (sz, sy, sx)), 0, 1)
    # ruido de camera (shot + leitura) na resolucao nativa
    pe = float(os.environ.get("SYNTH_POISSON_PE", "300")); rd = float(os.environ.get("SYNTH_READNOISE", "0.006"))
    vol = np.clip(rng.poisson(np.clip(vol, 0, 1)*pe)/pe + rng.normal(0, rd, vol.shape), 0, 1).astype(np.float32)

    # POOL 4x em XY. ATENCAO -- INCONSISTENCIA REAL CORRIGIDA: este gerador poolava por MEDIA, mas o
    # pipeline que PONTUA (`dog_max_infer.pool_xy`) poola por STRIDE (`[::1,::4,::4]`). Treinar/medir
    # num pooling e' fazer deploy noutro: a media suaviza o ruido (SNR maior) e o stride nao, entao o
    # dado sintetico chegava mais limpo do que o detector realmente ve. Default = STRIDE (= deploy).
    if os.environ.get("SYNTH_POOL_MODE", "stride") == "mean":
        vol_p = vol[:, :(Y//POOL)*POOL, :(X//POOL)*POOL].reshape(Z, Y//POOL, POOL, X//POOL, POOL).mean((2, 4)).astype(np.float32)
    else:
        vol_p = vol[::1, ::POOL, ::POOL].astype(np.float32)
    cens_p = (np.array(cens, np.float32) / np.array([1, POOL, POOL])) if cens else np.zeros((0, 3))
    return vol, vol_p, np.array(cens, np.float32), cens_p


# ================================================================================================
# VERBATIM cell 15, lines 5-22 (_to_u16, _real_stats)
# ================================================================================================

def _to_u16(v):
    """float [0,1] -> uint16. Halves the size versus float32 with no visible loss (the real data is
    uint16 as well)."""
    return np.clip(v * 65535.0, 0, 65535).astype(np.uint16)


def _real_stats(td, vids, n_frames=2):
    """REAL per-field count distribution + tissue-shell geometry (drives the generative placement)."""
    counts, pts = [], []
    for name in vids:
        for t in range(n_frames):
            try: vp = pool_xy(norm(read_frame(td/f"{name}.zarr", t)[0]))            # noqa: F821
            except Exception: continue
            d = dog_detect_precise(vp)[0]                                           # noqa: F821
            counts.append(len(d))
            if len(d) and len(pts) < 8: pts.append(np.atleast_2d(d)*np.array([1, POOL, POOL]))  # noqa: F821
    shell = fit_shell(np.concatenate(pts)) if pts else None                         # noqa: F821
    return np.array(counts, float), shell


# ================================================================================================
# ADAPTED cell 15 run_dataset_build(): calibration + ONE sequence (loop body of part B)
# ================================================================================================

STEP_MED, PERSIST, SISTER = 1.86, 0.30, 7.24
VOX = np.array([1.625, 0.40625, 0.40625])
s_step = (STEP_MED/1.5382)*1.10


def calibration_videos(td):
    vids = sorted(p.stem for p in td.glob("*.zarr")); by = {}
    for v in vids: by.setdefault(v.split("_")[0], []).append(v)
    embs = sorted(by)
    # Calibrate on one embryo only, leaving the other free for the user's own validation.
    calib = sorted(by[embs[0]])[:10]
    return calib


def calibrate(td=None):
    td = find_train_dir() if td is None else Path(td); assert td, "train directory not found"
    calib = calibration_videos(td)
    counts, shell = _real_stats(td, calib)
    return calib, counts, shell


def _rng(seed, base):
    """seed == 0 -> exactly the notebook's np.random.default_rng(base); otherwise an independent stream."""
    return np.random.default_rng(base) if int(seed) == 0 else np.random.default_rng([int(seed), int(base)])


def generate_sequence(j, counts, shell, T_SEQ=6, div_p=0.05, seed=0):
    """Returns dict(volumes, nodes, edges, divisions, voxel_um_pooled) or None when the notebook would
    have skipped this index (`if len(pos) < 10: j += 1; continue`)."""
    rg = _rng(seed, 500000+j)
    n0 = int(max(20, rg.choice(counts)*1.25))
    pos, radii = sample_shell_positions(n0, (64, 256, 256), rg, shell=shell,
                                        r_lo=3.5, r_hi=5.5)
    if len(pos) < 10: return None
    # per-cell speed scale (log-normal) reproduces the heavy tail of the real step distribution
    sc = np.exp(rg.normal(0, 0.55, len(pos)))
    vel = rg.normal(0, s_step, (len(pos), 3)) * sc[:, None]
    flow = rg.normal(0, 1, 3); flow /= np.linalg.norm(flow)+1e-9   # collective tissue drift
    tracks = [dict(pos=p.copy(), vel=vel[k], sc=sc[k], tid=k) for k, p in enumerate(pos)]
    nodes, edges, divs = [], [], []
    vols = np.zeros((T_SEQ, 64, 64, 64), np.uint16)                # pooled, to fit the budget
    nid = 0; prev_ids = {}
    for t in range(T_SEQ):
        cens = np.array([tr["pos"] for tr in tracks], np.float32)
        nat, _, cc, _ = gen_volume_native(zshape=64, xyshape=256, cens_native=cens,
                                          rng=_rng(seed, 900000+j*97+t), shell=shell)
        vols[t] = _to_u16(pool_xy(nat))
        cur_ids = {}
        for k, tr in enumerate(tracks):
            nodes.append([t, tr["pos"][0], tr["pos"][1], tr["pos"][2], tr["tid"]])
            cur_ids[k] = nid; nid += 1
        for k, tr in enumerate(tracks):
            if tr.get("parent_slot") is not None and tr["parent_slot"] in prev_ids:
                edges.append([prev_ids[tr["parent_slot"]], cur_ids[k]])
                if tr.get("is_div"): divs.append(prev_ids[tr["parent_slot"]])
        prev_ids = cur_ids
        if t == T_SEQ-1: break
        nxt = []
        for k, tr in enumerate(tracks):
            # Ornstein-Uhlenbeck step: persistent velocity + collective flow
            v = PERSIST*tr["vel"] + np.sqrt(1-PERSIST**2)*rg.normal(0, s_step*tr["sc"], 3)
            v = v + 0.35*s_step*flow
            if rg.random() < div_p:                                # DIVISION
                d = rg.normal(0, 1, 3); d /= np.linalg.norm(d)+1e-9
                half = 0.5*max(2.0, rg.normal(SISTER, 1.6))*d/VOX
                for sgn in (1, -1):
                    p2 = np.clip(tr["pos"]+v+sgn*half, [4, 12, 12], [59, 243, 243])
                    nxt.append(dict(pos=p2, vel=v*0.5, sc=tr["sc"], tid=tr["tid"],
                                    parent_slot=k, is_div=True))
            else:
                p2 = np.clip(tr["pos"]+v, [4, 12, 12], [59, 243, 243])
                nxt.append(dict(pos=p2, vel=v, sc=tr["sc"], tid=tr["tid"],
                                parent_slot=k, is_div=False))
        tracks = nxt
    return dict(volumes=vols, nodes=np.array(nodes, np.float32),
                edges=np.array(edges, np.int32), divisions=np.array(sorted(set(divs)), np.int32),
                voxel_um_pooled=np.array([1.625, 1.625, 1.625], np.float32))

