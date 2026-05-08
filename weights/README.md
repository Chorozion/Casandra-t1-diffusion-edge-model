# Cassandra T1 Weights

This directory contains the open-release Cassandra T1 checkpoint artifacts tracked through Git LFS.

## Included Checkpoints

| File | Source | Status | Size | SHA256 |
|---|---|---|---:|---|
| File parts | Source | Status | Reassembled size | Reassembled SHA256 |
|---|---|---|---:|---|
| `cassandra_ep5_fp16.pt.part001-002` | `I:\sophiat1\checkpoints\epoch5\cassandra_ep5_fp16.pt` | Verified epoch-5 FP16 checkpoint | 2,659,500,664 bytes | `D70C813C513F5232A25313FA60338F862020BA942ED26D54F62511766FA5F044` |
| `v2_scratch_epoch2_82002.pt.part001-009` | `I:\sophiat1\backup\runpod_20260422\v2_scratch_epoch2_82002.pt` | Latest checkpoint found by timestamp in the source directory | 16,665,974,950 bytes | `8BFB5644209AB8FD241D85A725781495B29922811A2BECF8260AAAAD0A26DA6F` |

## Notes

- `cassandra_ep5_fp16.pt` is the checkpoint used by the local and server inference scripts in this release after reassembly.
- `v2_scratch_epoch2_82002.pt` is included because it is the newest checkpoint artifact found in `I:\sophiat1`.
- Newer is not automatically better. The source comparison file indicates the v2 scratch output was still unstable.
- These files are research checkpoints and should be evaluated before any production use.

## Reassembly

PowerShell:

```powershell
Get-Content .\cassandra_ep5_fp16.pt.part* -Encoding Byte -ReadCount 0 | Set-Content .\cassandra_ep5_fp16.pt -Encoding Byte
Get-Content .\v2_scratch_epoch2_82002.pt.part* -Encoding Byte -ReadCount 0 | Set-Content .\v2_scratch_epoch2_82002.pt -Encoding Byte
```

Bash:

```bash
cat cassandra_ep5_fp16.pt.part* > cassandra_ep5_fp16.pt
cat v2_scratch_epoch2_82002.pt.part* > v2_scratch_epoch2_82002.pt
```

Use `checksums.sha256` to verify each part before reassembly.
