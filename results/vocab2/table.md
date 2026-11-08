
### style A: 18 clean labels of 18
| Model | all 18 | clean | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|---|
| ru-jev-v1.1 | 0.856 | 0.856 | 0.852 | -0.004 | 28 |
| ru-jev-v1.3 | 0.889 | 0.889 | 0.882 | -0.007 | 36 |
| laya instr A | 0.557 | 0.557 | 0.671 | +0.114 | 59 |

Questions 1000, candidates 5, chance 0.200.

### style B: 17 clean labels of 18

Excluded from the clean figure (exact match in training): recommendation

| Model | all 18 | clean | Mean confidence | Gap | Confident errors per 1000 |
|---|---|---|---|---|---|
| ru-jev-v1.1 | 0.403 | 0.420 | 0.580 | +0.177 | 76 |
| ru-jev-v1.3 | 0.379 | 0.360 | 0.643 | +0.264 | 114 |
| laya instr A | 0.322 | 0.331 | 0.502 | +0.180 | 40 |

Questions 1000, candidates 5, chance 0.200.

### style A: accuracy per label

| Label | surface form | n | v1.1 | v1.3 | laya | v1.3 minus v1.1 |
|---|---|---|---|---|---|---|
| recommendation | RECOMMENDATION_SCN | 41 | 1.000 | 0.927 | 0.512 | -0.073 |
| music | MUSIC_SCN | 20 | 1.000 | 0.950 | 0.800 | -0.050 |
| audio | AUDIO_SCN | 22 | 0.955 | 0.909 | 0.409 | -0.045 |
| cooking | COOKING_SCN | 22 | 0.909 | 0.864 | 0.909 | -0.045 |
| qa | QA_SCN | 106 | 0.547 | 0.538 | 0.283 | -0.009 |
| calendar | CALENDAR_SCN | 127 | 0.961 | 0.953 | 0.504 | -0.008 |
| email | EMAIL_SCN | 86 | 0.977 | 0.977 | 0.907 | +0.000 |
| lists | LISTS_SCN | 58 | 0.983 | 0.983 | 0.483 | +0.000 |
| news | NEWS_SCN | 39 | 0.974 | 0.974 | 0.718 | +0.000 |
| transport | TRANSPORT_SCN | 47 | 1.000 | 1.000 | 0.872 | +0.000 |
| general | GENERAL_SCN | 68 | 0.868 | 0.882 | 0.059 | +0.015 |
| weather | WEATHER_SCN | 57 | 0.982 | 1.000 | 0.754 | +0.018 |
| play | PLAY_SCN | 135 | 0.941 | 0.978 | 0.681 | +0.037 |
| datetime | DATETIME_SCN | 35 | 0.886 | 0.971 | 0.829 | +0.086 |
| social | SOCIAL_SCN | 33 | 0.848 | 0.939 | 0.758 | +0.091 |
| alarm | ALARM_SCN | 31 | 0.516 | 0.613 | 0.387 | +0.097 |
| takeaway | TAKEAWAY_SCN | 24 | 0.500 | 0.792 | 0.125 | +0.292 |
| iot | IOT_SCN | 49 | 0.388 | 0.755 | 0.286 | +0.367 |

Where v1.3's errors go: general 28, play 16, news 13, transport 8, iot 7, email 6

### style B: accuracy per label

| Label | surface form | n | v1.1 | v1.3 | laya | v1.3 minus v1.1 |
|---|---|---|---|---|---|---|
| music | МУЗ | 20 | 0.650 | 0.050 | 0.400 | -0.600 |
| lists | СПИ | 58 | 0.552 | 0.017 | 0.655 | -0.534 |
| qa | СПР | 106 | 0.689 | 0.198 | 0.179 | -0.491 |
| calendar | КАЛ | 127 | 0.331 | 0.102 | 0.181 | -0.228 |
| play | ВОС | 135 | 0.200 | 0.007 | 0.148 | -0.193 |
| cooking | КУЛ | 22 | 0.909 | 0.864 | 0.500 | -0.045 |
| datetime | ДАТ | 35 | 0.943 | 0.943 | 0.800 | +0.000 |
| transport | ТРА | 47 | 0.277 | 0.362 | 0.170 | +0.085 |
| general | ОБЩ | 68 | 0.588 | 0.676 | 0.029 | +0.088 |
| audio | ЗВУ | 22 | 0.909 | 1.000 | 0.455 | +0.091 |
| iot | УМН | 49 | 0.184 | 0.286 | 0.367 | +0.102 |
| email | ПОЧ | 86 | 0.279 | 0.419 | 0.360 | +0.140 |
| social | СОЦ | 33 | 0.636 | 0.788 | 0.576 | +0.152 |
| news | НОВ | 39 | 0.615 | 0.846 | 0.462 | +0.231 |
| weather | ПОГ | 57 | 0.070 | 0.351 | 0.474 | +0.281 |
| takeaway | ДОС | 24 | 0.250 | 0.542 | 0.417 | +0.292 |
| recommendation (exact match in training) | РЕК | 41 | 0.000 | 0.829 | 0.122 | +0.829 |
| alarm | БУД | 31 | 0.065 | 0.935 | 0.871 | +0.871 |

Where v1.3's errors go: qa 78, transport 72, news 65, weather 59, email 57, takeaway 51
