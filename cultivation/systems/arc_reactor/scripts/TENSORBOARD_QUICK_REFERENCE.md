# TensorBoard Quick Reference

## 🚀 Quick Start Commands

```bash
# First time setup (run once)
./setup_tensorboard_env.sh

# Launch TensorBoard
./run_tensorboard.sh

# Open browser
# http://localhost:6006

# Stop TensorBoard
# Press Ctrl+C or: pkill -f tensorboard
```

## 📊 Your ARC Reactor Results

| Version | Test Cell Acc | Test Grid Acc | Status |
|---------|---------------|---------------|---------|
| version_1 | **83.5%** | **33.7%** | ✅ Best |
| version_2 | 68.2% | 15.9% | ⚠️ Lower |
| version_0 | 0.0% | 0.0% | 🔍 Test run |

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| TensorBoard won't start | `./setup_tensorboard_env.sh` |
| Port 6006 in use | `pkill -f tensorboard` |
| No data visible | Check logs exist in `cultivation/systems/arc_reactor/logs/training/lightning_logs/` |
| Environment broken | `rm -rf .tensorboard_env && ./setup_tensorboard_env.sh` |

## 📁 File Locations

- **Setup Script**: `setup_tensorboard_env.sh`
- **Launch Script**: `run_tensorboard.sh`
- **Environment**: `.tensorboard_env/`
- **Training Logs**: `cultivation/systems/arc_reactor/logs/training/lightning_logs/`
- **Full Documentation**: `docs/tensorboard_setup_guide.md`

## 🎯 Key Metrics to Watch

- `train_loss` / `val_loss_epoch` - Should decrease
- `train_cell_accuracy` / `val_cell_accuracy` - Should increase
- `train_grid_accuracy` / `val_grid_accuracy` - Should increase
- Gap between train/val - Should be reasonable (not too large)

---
📖 **Full Guide**: See `docs/tensorboard_setup_guide.md` for complete documentation
