# SP100 Diagnostic Summary

## kappa_response
- mean=1.427, corr(vol)=0.819, corr(dd)=0.779, crisis2020=1.844 vs normal=1.265
- supports: unclear
- next: 若 corr 为负则检查 beta 符号；若 mean 过高则降低 kappa_max 或平滑

## weight_concentration
- C eff_n=70.3 vs B 79.4 vs A 87.7; C max_w=0.106
- supports: yes
- next: 若 C 更集中：加 HHI penalty 或单资产上限

## turnover
- {'A_no_kappa': 0.2721629786762381, 'B_fixed_kappa': 0.3349466022995986, 'C_manual_kappa': 0.6065326001399092, 'D_state_action': 0.4686360315571219, 'Historical_CVaR': 0.2721629786762381}
- supports: yes
- next: 若 C 换手更高：平滑 kappa 或降低调仓频率

## sector_exposure
- [{'method': 'C_manual_kappa', 'sector_hhi_mean': 0.2067298898153641, 'max_sector_exposure_mean': 0.2800120744666996, 'max_sector_exposure_max': 0.517165871195554, 'top_sector_at_max': '2021-02-26'}, {'method': 'B_fixed_kappa', 'sector_hhi_mean': 0.1942744802857249, 'max_sector_exposure_mean': 0.2617157193829227, 'max_sector_exposure_max': 0.5770267526410118, 'top_sector_at_max': '2021-03-31'}, {'method': 'Historical_CVaR', 'sector_hhi_mean': 0.1887502785196323, 'max_sector_exposure_mean': 0.265553916325919, 'max_sector_exposure_max': 0.7321227447236442, 'top_sector_at_max': '2020-04-30'}]
- supports: unclear
- next: 若行业 HHI 高：加 sector cap

## q_concentration
- [{'method': 'A_no_kappa', 'q_hhi_mean': 0.0778533635676492, 'top1_q_mean': 0.0793650793650793, 'top3_q_mean': 0.238095238095238, 'top5_q_mean': 0.3968253968253967}, {'method': 'B_fixed_kappa', 'q_hhi_mean': 0.1534391534391534, 'top1_q_mean': 0.1587301587301587, 'top3_q_mean': 0.4761904761904761, 'top5_q_mean': 0.7936507936507934}, {'method': 'C_manual_kappa', 'q_hhi_mean': 0.1292725640137372, 'top1_q_mean': 0.1507714259762336, 'top3_q_mean': 0.4317769661070778, 'top5_q_mean': 0.6910135598592773}]
- supports: unclear
- next: 若 q 更集中：增大 M 或降低 kappa_max

## kappa_max_sensitivity
- best CVaR=0.0256 at kappa_max=1.25
- supports: yes
- next: 状态依赖机制可能无效

## window_sensitivity
- best CVaR=0.0281 at M=504
- supports: yes
- next: 增大估计窗口 M
