# VisionGuard experiments. Each run differs from mnv2_baseline by ONE change.
function Invoke-Run {
    param([string[]]$RunArgs)
    Write-Host "`n===== $($RunArgs -join ' ') =====" -ForegroundColor Cyan
    python src\run_experiment.py @RunArgs
    if ($LASTEXITCODE -ne 0) { throw "Run failed: $($RunArgs -join ' ')" }
}

# 3.3: same CNN, manual tracking vs autolog
Invoke-Run @("--model","cnn","--run-name","cnn_baseline","--description","Backbone: scratch CNN")
Invoke-Run @("--model","cnn","--run-name","cnn_baseline_autolog","--tracking","autolog","--epochs","15","--description","Autolog demo")

# 3.5: transfer baseline + one-change variations
Invoke-Run @("--model","transfer","--run-name","mnv2_baseline","--description","Reference run")
Invoke-Run @("--model","transfer","--run-name","mnv2_no_augment","--no-augment","--description","Change: augmentation off")
Invoke-Run @("--model","transfer","--run-name","mnv2_ftlr_1e-4","--ft-lr","1e-4","--description","Change: fine-tune LR 1e-5 -> 1e-4")
Invoke-Run @("--model","transfer","--run-name","mnv2_unfreeze_3blocks","--unfreeze-blocks","3","--description","Change: unfreeze last 3 blocks")

$env:RESIZE_MODE = "pad"
Invoke-Run @("--model","transfer","--run-name","mnv2_pad","--description","Change: resize mode crop -> pad")
Remove-Item Env:RESIZE_MODE

$env:IMG_SIZE = "224"
Invoke-Run @("--model","transfer","--run-name","mnv2_img224","--description","Change: image size 160 -> 224")
Remove-Item Env:IMG_SIZE

# Optional: best settings from the 2.7 tuner (fill in your values)
# Invoke-Run @("--model","transfer","--run-name","mnv2_tuned","--ft-lr","<x>","--dropout","<x>","--unfreeze-blocks","<x>","--description","Keras Tuner best")
Write-Host "`nAll runs finished." -ForegroundColor Green