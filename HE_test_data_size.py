
# runner.py
import os, json, uuid, time, platform, pickle
from datetime import datetime
import numpy as np
import pandas as pd

# Try TenSEAL import and fail fast with a helpful message
try:
    import tenseal as ts
except Exception as e:
    raise RuntimeError(
        "TenSEAL is required for CKKS/BFV. Install it in your venv:\n"
        "  python -m venv .venv && source .venv/bin/activate\n"
        "  pip install numpy pandas matplotlib pyyaml tenseal torch\n"
        f"Original import error: {repr(e)}"
    )

# -------------------- General utils --------------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def tms(start_ns, end_ns):
    return (end_ns - start_ns) / 1e6  # ms

def sys_info():
    return {
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }

def simulate_transport(bytes_count: int, bandwidth_mbps: float) -> float:
    # ms = (8 * bytes) / (Mbps * 1e6) * 1e3
    return (8.0 * bytes_count) / (bandwidth_mbps * 1e6) * 1e3

# -------------------- Payload generators --------------------
def make_image(size_label: str, dtype=np.float32, seed=0):
    H, W = map(int, size_label.split("x"))
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((1,1,H,W), dtype=dtype)
    return arr

def make_vector(L: int, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((L,), dtype=dtype)

def make_gradient(L: int, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((L,), dtype=dtype)

def flatten_payload(x: np.ndarray):
    return x.reshape(-1).copy()

# -------------------- (De)serialization helpers --------------------
def serialize(obj, fmt="pickle"):
    t0 = time.perf_counter_ns()
    if fmt == "pickle":
        b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "library-native":
        # Expect bytes from library .serialize()
        if isinstance(obj, (bytes, bytearray)):
            b = obj
        else:
            # Fallback for unexpected type
            b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unknown serialize format: {fmt}")
    t1 = time.perf_counter_ns()
    return b, tms(t0, t1)

def deserialize(b, fmt="pickle"):
    t0 = time.perf_counter_ns()
    if fmt == "pickle":
        obj = pickle.loads(b)
    elif fmt == "library-native":
        # Caller should wrap library-native blobs appropriately
        obj = b
    else:
        raise ValueError(f"Unknown serialize format: {fmt}")
    t1 = time.perf_counter_ns()
    return obj, tms(t0, t1)

# -------------------- TenSEAL helpers --------------------
def ckks_max_slots(poly_degree: int) -> int:
    # CKKS slots = poly_degree / 2
    return poly_degree // 2

def chunk_iter(arr: np.ndarray, chunk_size: int):
    for i in range(0, arr.size, chunk_size):
        yield arr[i:i+chunk_size]

# -------------------- HE: CKKS --------------------
class HECKKS:
    def __init__(self, poly_degree: int, scale: float, coeff_mod_bits: list[int]):
        self.params = dict(poly_degree=poly_degree, scale=scale, coeff_mod_bits=coeff_mod_bits)
        t0 = time.perf_counter_ns()
        # Secret context (server/decryptor)
        self.ctx_secret = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_degree,
            coeff_mod_bit_sizes=coeff_mod_bits
        )
        self.ctx_secret.generate_galois_keys()
        self.ctx_secret.generate_relin_keys()
        self.ctx_secret.global_scale = scale
        # Public context (client/encryptor): remove secret key
        self.ctx_public = ts.context_from(self.ctx_secret.serialize())
        self.ctx_public.make_context_public()
        t1 = time.perf_counter_ns()
        self.t_keygen_ms = tms(t0, t1)

    def encode_pack(self, vec: np.ndarray, batch_size=1):
        # Pack up to slots-per-ciphertext
        slots = ckks_max_slots(self.params["poly_degree"])
        effective_chunk = min(slots, vec.size)
        t0 = time.perf_counter_ns()
        chunks = [v.copy() for v in chunk_iter(vec, effective_chunk)]
        t1 = time.perf_counter_ns()
        return chunks, tms(t0, t1)

    def encrypt(self, chunks: list[np.ndarray]):
        cts, t_enc_ms = [], 0.0
        for ch in chunks:
            t0 = time.perf_counter_ns()
            ct = ts.ckks_vector(self.ctx_public, ch.tolist())
            t1 = time.perf_counter_ns()
            cts.append(ct)
            t_enc_ms += tms(t0, t1)
        return cts, t_enc_ms

    def serialize_ciphertexts(self, cts):
        sizes, total_ms, blobs = [], 0.0, []
        for ct in cts:
            t0 = time.perf_counter_ns()
            b = ct.serialize()
            t1 = time.perf_counter_ns()
            sizes.append(len(b))
            total_ms += tms(t0, t1)
            blobs.append(b)
        return blobs, sizes, total_ms

    def deserialize_ciphertexts(self, blobs):
        cts, total_ms = [], 0.0
        for b in blobs:
            t0 = time.perf_counter_ns()
            ct = ts.ckks_vector_from(self.ctx_public, b)
            t1 = time.perf_counter_ns()
            cts.append(ct)
            total_ms += tms(t0, t1)
        return cts, total_ms

    def decrypt(self, cts):
        rec, t_dec_ms = [], 0.0
        for ct in cts:
            # Re-serialize to load with secret context
            blob = ct.serialize()
            t0 = time.perf_counter_ns()
            vals = ts.ckks_vector_from(self.ctx_secret, blob).decrypt()
            t1 = time.perf_counter_ns()
            rec.extend(vals)
            t_dec_ms += tms(t0, t1)
        return np.array(rec), t_dec_ms

# -------------------- HE: BFV --------------------
class HEBFV:
    def __init__(self, poly_degree: int, t_bits: int):
        self.params = dict(poly_degree=poly_degree, plaintext_modulus_bits=t_bits)
        plaintext_modulus = 2 ** t_bits
        t0 = time.perf_counter_ns()
        self.ctx_secret = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=poly_degree,
            plain_modulus=plaintext_modulus
        )
        self.ctx_secret.generate_galois_keys()
        self.ctx_public = ts.context_from(self.ctx_secret.serialize())
        self.ctx_public.make_context_public()
        t1 = time.perf_counter_ns()
        self.t_keygen_ms = tms(t0, t1)

    def _quantize_to_int(self, vec: np.ndarray):
        # Quantize floats to integers for BFV (simple scaling)
        scale = 1000.0
        q = np.clip(np.rint(vec * scale),
                    - (2**15), (2**15)-1).astype(np.int64)
        return q, scale

    def encode_pack(self, vec: np.ndarray, batch_size=1):
        # Pack up to poly_degree integers per ciphertext
        slots = self.params["poly_degree"]
        q, scale = self._quantize_to_int(vec)
        t0 = time.perf_counter_ns()
        chunks = [q[i:i+slots].copy() for i in range(0, q.size, slots)]
        t1 = time.perf_counter_ns()
        self._scale = scale
        return chunks, tms(t0, t1)

    def encrypt(self, chunks: list[np.ndarray]):
        cts, t_enc_ms = [], 0.0
        for ch in chunks:
            t0 = time.perf_counter_ns()
            ct = ts.bfv_vector(self.ctx_public, ch.tolist())
            t1 = time.perf_counter_ns()
            cts.append(ct)
            t_enc_ms += tms(t0, t1)
        return cts, t_enc_ms

    def serialize_ciphertexts(self, cts):
        sizes, total_ms, blobs = [], 0.0, []
        for ct in cts:
            t0 = time.perf_counter_ns()
            b = ct.serialize()
            t1 = time.perf_counter_ns()
            sizes.append(len(b))
            total_ms += tms(t0, t1)
            blobs.append(b)
        return blobs, sizes, total_ms

    def deserialize_ciphertexts(self, blobs):
        cts, total_ms = [], 0.0
        for b in blobs:
            t0 = time.perf_counter_ns()
            ct = ts.bfv_vector_from(self.ctx_public, b)
            t1 = time.perf_counter_ns()
            cts.append(ct)
            total_ms += tms(t0, t1)
        return cts, total_ms

    def decrypt(self, cts):
        rec, t_dec_ms = [], 0.0
        for ct in cts:
            blob = ct.serialize()
            t0 = time.perf_counter_ns()
            vals = ts.bfv_vector_from(self.ctx_secret, blob).decrypt()  # list[int]
            t1 = time.perf_counter_ns()
            rec.extend(vals)
            t_dec_ms += tms(t0, t1)
        arr = np.array(rec, dtype=np.float64) / getattr(self, "_scale", 1.0)
        return arr, t_dec_ms

# -------------------- SMPC (NumPy additive sharing) --------------------
class SMPCAdditive:
    def __init__(self, parties=5, dropout_rate=0.0):
        self.params = dict(parties=parties, dropout_rate=dropout_rate)

    def share_generate(self, vec: np.ndarray):
        rng = np.random.default_rng()
        t0 = time.perf_counter_ns()
        shares = [rng.standard_normal(vec.shape, dtype=np.float64)
                  for _ in range(self.params["parties"] - 1)]
        last = vec.astype(np.float64) - np.sum(shares, axis=0)
        shares.append(last)
        t1 = time.perf_counter_ns()
        return shares, tms(t0, t1)

    def serialize_shares(self, shares, fmt="pickle"):
        sizes, total_ms, blobs = [], 0.0, []
        for sh in shares:
            b, t_ser = serialize(sh, fmt=fmt)
            sizes.append(len(b)); total_ms += t_ser; blobs.append(b)
        return blobs, sizes, total_ms

    def deserialize_shares(self, blobs, fmt="pickle"):
        shares, total_ms = [], 0.0
        for b in blobs:
            sh, t_des = deserialize(b, fmt=fmt)
            shares.append(sh); total_ms += t_des
        return shares, total_ms

    def reconstruct(self, shares: list[np.ndarray]):
        t0 = time.perf_counter_ns()
        rec = np.sum(shares, axis=0)
        t1 = time.perf_counter_ns()
        return rec, tms(t0, t1)

# -------------------- Trial runner --------------------
def run_trial(modality, size_label, crypto, batch_size, precision,
              serialize_fmt, bandwidths, repeat_idx, seed):
    # 1) payload
    if modality == "image":
        arr = make_image(size_label, dtype=np.float32 if precision=="float32" else np.float64, seed=seed)
        vec = flatten_payload(arr)
    elif modality == "vector":
        L = int(size_label)
        vec = make_vector(L, dtype=np.float32 if precision=="float32" else np.float64, seed=seed)
    elif modality == "gradient":
        L = int(size_label)
        vec = make_gradient(L, dtype=np.float32 if precision=="float32" else np.float64, seed=seed)
    else:
        raise ValueError("Unknown modality")

    bytes_plain = vec.nbytes
    t_keygen = None

    if crypto["kind"] == "HE-CKKS":
        he = HECKKS(crypto["poly_degree"], crypto["scale"], crypto["coeff_mod_bits"])
        t_keygen = he.t_keygen_ms
        chunks, t_encode = he.encode_pack(vec, batch_size=batch_size)
        cts, t_encrypt = he.encrypt(chunks)
        blobs, sizes_list, t_ser = he.serialize_ciphertexts(cts)
        bytes_cipher_or_shares = int(sum(sizes_list))
        cts2, t_deser = he.deserialize_ciphertexts(blobs)
        dec, t_decrypt = he.decrypt(cts2)
        vec_cmp = vec[:dec.size]
        mae = float(np.mean(np.abs(dec - vec_cmp)))
        exact = False

    elif crypto["kind"] == "HE-BFV":
        he = HEBFV(crypto["poly_degree"], crypto["t_bits"])
        t_keygen = he.t_keygen_ms
        chunks, t_encode = he.encode_pack(vec, batch_size=batch_size)
        cts, t_encrypt = he.encrypt(chunks)
        blobs, sizes_list, t_ser = he.serialize_ciphertexts(cts)
        bytes_cipher_or_shares = int(sum(sizes_list))
        cts2, t_deser = he.deserialize_ciphertexts(blobs)
        dec, t_decrypt = he.decrypt(cts2)
        vec_cmp = vec[:dec.size]
        exact = bool(np.allclose(dec, vec_cmp, atol=1e-12))
        mae = float(np.mean(np.abs(dec - vec_cmp)))

    elif crypto["kind"] == "SMPC-additive":
        s = SMPCAdditive(parties=crypto["parties"], dropout_rate=crypto["dropout"])
        t_encode = 0.0
        shares, t_share = s.share_generate(vec)
        blobs, sizes_list, t_ser = s.serialize_shares(shares, fmt=serialize_fmt)
        bytes_cipher_or_shares = int(sum(sizes_list))
        dshares, t_deser = s.deserialize_shares(blobs, fmt=serialize_fmt)
        rec, t_recon = s.reconstruct(dshares)
        t_encrypt, t_decrypt = t_share, t_recon
        mae = float(np.mean(np.abs(rec - vec)))
        exact = bool(np.allclose(rec, vec, atol=0.0))

    else:
        raise ValueError("Unknown crypto kind")

    # 6) optional transport
    bytes_over_network = bytes_cipher_or_shares
    t_network = {f"{bw}Mbps": simulate_transport(bytes_over_network, bw) for bw in bandwidths}

    record = {
        "trial_id": str(uuid.uuid4()),
        "timestamp": now_iso(),
        "modality": modality,
        "size_label": size_label,
        "crypto": crypto["kind"],
        "batch_size": batch_size,
        "precision": precision,
        "params": {k:v for k,v in crypto.items() if k!="kind"},
        "bytes_plain": bytes_plain,
        "bytes_encrypted_or_shares": bytes_cipher_or_shares,
        "bytes_over_network": bytes_over_network,
        "timing_ms": {
            "t_keygen": t_keygen or 0.0,
            "t_encode": t_encode,
            "t_encrypt_or_share": t_encrypt,
            "t_serialize": t_ser,
            "t_deserialize": t_deser,
            "t_decrypt_or_reconstruct": t_decrypt,
            "t_aggregate": 0.0
        },
        "quality": {
            "mae": mae,
            "mse": None,
            "exact_match": exact
        },
        "system": sys_info(),
        "repeat_index": repeat_idx
    }
    record["derived"] = {
        "time_per_element_ms": (record["timing_ms"]["t_encrypt_or_share"] / max(1, vec.size)),
        "overhead_ratio": (bytes_cipher_or_shares / max(1, bytes_plain)),
        "network_time_ms": t_network
    }
    return record

# -------------------- Grid expansion --------------------
def grid_configs(cfg):
    for modality, sdict in cfg["modalities"].items():
        if modality == "image":
            size_list = sdict["sizes"]
        else:
            size_list = list(map(str, sdict["sizes"]))

        # HE-CKKS
        for d in cfg["crypto"]["he_ckks"]["poly_degrees"]:
            for scale in cfg["crypto"]["he_ckks"]["scales"]:
                for modbits in cfg["crypto"]["he_ckks"]["coeff_modulus_bits"]:
                    for bs in cfg["crypto"]["he_ckks"]["batch_sizes"]:
                        yield modality, size_list, {"kind":"HE-CKKS","poly_degree":d,"scale":scale,"coeff_mod_bits":modbits}, bs

        # HE-BFV
        for d in cfg["crypto"]["he_bfv"]["poly_degrees"]:
            for t_bits in cfg["crypto"]["he_bfv"]["plaintext_moduli_bits"]:
                for bs in cfg["crypto"]["he_bfv"]["batch_sizes"]:
                    yield modality, size_list, {"kind":"HE-BFV","poly_degree":d,"t_bits":t_bits}, bs

        # SMPC
        for p in cfg["crypto"]["smpc_additive"]["parties"]:
            for dr in cfg["crypto"]["smpc_additive"]["dropout_rates"]:
                yield modality, size_list, {"kind":"SMPC-additive","parties":p,"dropout":dr}, 1

# -------------------- Main --------------------
def main():
    # Load config.yaml if present
    cfg_path = "config.yaml"
    if os.path.exists(cfg_path):
        import yaml
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "experiment":{"name":"he_smpc_benchmark","seed":20260103,"num_repeats":5,"serialize_formats":["library-native","pickle"],"bandwidth_mbps":[10,100],"threads":1},
            "modalities":{
                "image":{"sizes":["64x64","128x128","256x256"],"dtype":"float32"},
                "vector":{"sizes":[32,128,512,1024,4096,16384],"dtype":"float32"},
                "gradient":{"sizes":[10000,50000,100000,500000],"dtype":"float32"}
            },
            "crypto":{
                "he_ckks":{"poly_degrees":[16384,32768],
                           "scales":[33554432,34359738368], # 2^25, 2^35
                           "coeff_modulus_bits":[[60,40,40,60],[60,60,40,60]],
                           "batch_sizes":[1,8,32,128]},
                "he_bfv":{"poly_degrees":[8192,16384],
                          "plaintext_moduli_bits":[15,16],
                          "batch_sizes":[1,8,32,128]},
                "smpc_additive":{"parties":[3,5,10],
                                 "dropout_rates":[0.0,0.1]}
            },
            "precision":{"encode":["float32","float64"]}
        }

    out_dir = os.path.join("results")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    raw_path = os.path.join(out_dir, "raw_trials.csv")
    manifest_path = os.path.join(out_dir, "manifest.json")

    records = []
    rng = np.random.default_rng(cfg["experiment"]["seed"])

    for modality, size_list, crypto, bs in grid_configs(cfg):
        for size_label in size_list:
            for prec in cfg["precision"]["encode"]:
                for fmt in cfg["experiment"]["serialize_formats"]:
                    for r in range(cfg["experiment"]["num_repeats"]):
                        seed = int(rng.integers(0, 2**31-1))
                        rec = run_trial(
                            modality=modality,
                            size_label=size_label,
                            crypto=crypto,
                            batch_size=bs,
                            precision=prec,
                            serialize_fmt=fmt,
                            bandwidths=cfg["experiment"]["bandwidth_mbps"],
                            repeat_idx=r,
                            seed=seed
                        )
                        records.append(rec)

    # Save raw trials
    df = pd.json_normalize(records)
    df.to_csv(raw_path, index=False)

    # Save manifest
    manifest = {
        "name": cfg["experiment"]["name"],
        "ts": now_iso(),
        "system": sys_info(),
        "n_trials": len(records)
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Aggregate summary
    group_cols = ["modality","size_label","crypto","batch_size","precision"]
    agg = df.groupby(group_cols).agg(
        bytes_plain=("bytes_plain","mean"),
        bytes_ciphertext_or_shares=("bytes_encrypted_or_shares","mean"),
        t_encode_ms_mean=("timing_ms.t_encode","mean"),
        t_encrypt_ms_mean=("timing_ms.t_encrypt_or_share","mean"),
        t_serialize_ms_mean=("timing_ms.t_serialize","mean"),
        t_decrypt_ms_mean=("timing_ms.t_decrypt_or_reconstruct","mean"),
        error_mae_mean=("quality.mae","mean"),
        repeats=("repeat_index","count"),
    ).reset_index()
    agg["system_tag"] = manifest["system"]["cpu"]
    agg["library_tag"] = "TenSEAL (CKKS/BFV), NumPy SMPC"
    agg.to_csv(os.path.join(out_dir, "summary_by_modality_size_crypto_batch.csv"), index=False)

    print(f"✅ Done. Raw: {raw_path}")
    print(f"✅ Summary: {os.path.join(out_dir, 'summary_by_modality_size_crypto_batch.csv')}")
    print(f"✅ Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
