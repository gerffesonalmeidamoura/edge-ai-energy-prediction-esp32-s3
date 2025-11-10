#pragma once
#include <stdint.h>
#include <stddef.h>

// ==== Decision Tree (header-only) ====
#define DT_INPUT_DIM 50

// Assinatura pública:
//   float dt_predict(const float f[DT_INPUT_DIM]);

// Implementação gerada a partir de modelo_decision_inferencia.c
// Torna a função inline para uso direto no firmware.

static inline float dt_predict(float f[]) {
    if (f[0] <= 9.584814) {
        if (f[2] <= -12.756489) {
            if (f[0] <= -37.125580) {
                if (f[2] <= -28.422316) {
                    if (f[3] <= 16.333167) {
                        if (f[8] <= 0.986037) {
                            if (f[8] <= -5.345465) {
                                if (f[48] <= -0.461648) {
                                    if (f[14] <= -2.999318) {
                                        return 24.790001;
                                    } else {
                                        if (f[41] <= 0.003315) {
                                            return 25.070000;
                                        } else {
                                            return 25.180000;
                                        }
                                    }
                                } else {
                                    return 25.770000;
                                }
                            } else {
                                if (f[29] <= 1.849629) {
                                    if (f[43] <= 0.629350) {
                                        if (f[27] <= 1.087247) {
                                            return 23.917500;
                                        } else {
                                            return 24.091667;
                                        }
                                    } else {
                                        if (f[7] <= 13.139529) {
                                            return 23.482500;
                                        } else {
                                            return 23.855000;
                                        }
                                    }
                                } else {
                                    if (f[30] <= 0.052973) {
                                        if (f[17] <= -2.083433) {
                                            return 24.559999;
                                        } else {
                                            return 24.500000;
                                        }
                                    } else {
                                        if (f[9] <= -2.747392) {
                                            return 24.635000;
                                        } else {
                                            return 24.730000;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f[28] <= 1.567400) {
                                if (f[14] <= 1.301711) {
                                    if (f[30] <= -0.058249) {
                                        if (f[41] <= 0.594096) {
                                            return 22.840000;
                                        } else {
                                            return 22.850000;
                                        }
                                    } else {
                                        if (f[22] <= 3.831677) {
                                            return 22.770000;
                                        } else {
                                            return 22.799999;
                                        }
                                    }
                                } else {
                                    return 23.440001;
                                }
                            } else {
                                if (f[10] <= 4.206340) {
                                    return 22.250000;
                                } else {
                                    return 22.270000;
                                }
                            }
                        }
                    } else {
                        if (f[7] <= -4.705143) {
                            if (f[38] <= 0.850269) {
                                if (f[46] <= -0.227413) {
                                    if (f[34] <= 0.638418) {
                                        if (f[16] <= 2.380263) {
                                            return 26.389999;
                                        } else {
                                            return 26.360001;
                                        }
                                    } else {
                                        return 26.540001;
                                    }
                                } else {
                                    if (f[17] <= -1.192822) {
                                        return 26.160000;
                                    } else {
                                        return 26.270000;
                                    }
                                }
                            } else {
                                return 27.139999;
                            }
                        } else {
                            if (f[6] <= 3.704422) {
                                if (f[24] <= -2.095896) {
                                    return 24.600000;
                                } else {
                                    if (f[29] <= 1.054277) {
                                        if (f[3] <= 25.760441) {
                                            return 25.320000;
                                        } else {
                                            return 25.180000;
                                        }
                                    } else {
                                        return 25.879999;
                                    }
                                }
                            } else {
                                return 23.440001;
                            }
                        }
                    }
                } else {
                    if (f[5] <= -5.853612) {
                        if (f[20] <= -1.406189) {
                            if (f[31] <= 0.293804) {
                                if (f[26] <= -1.158157) {
                                    return 27.150000;
                                } else {
                                    return 26.870001;
                                }
                            } else {
                                if (f[29] <= 0.319877) {
                                    return 26.450001;
                                } else {
                                    return 26.559999;
                                }
                            }
                        } else {
                            if (f[45] <= 0.671855) {
                                if (f[0] <= -47.628040) {
                                    return 28.219999;
                                } else {
                                    if (f[44] <= -0.419919) {
                                        return 27.809999;
                                    } else {
                                        return 27.940001;
                                    }
                                }
                            } else {
                                if (f[40] <= 0.482707) {
                                    return 27.250000;
                                } else {
                                    return 27.530001;
                                }
                            }
                        }
                    } else {
                        if (f[23] <= 1.056012) {
                            if (f[10] <= 0.274714) {
                                if (f[2] <= -26.268898) {
                                    return 24.040001;
                                } else {
                                    if (f[11] <= -0.944027) {
                                        return 24.459999;
                                    } else {
                                        return 24.350000;
                                    }
                                }
                            } else {
                                if (f[45] <= -0.527315) {
                                    return 25.870001;
                                } else {
                                    if (f[24] <= -1.778547) {
                                        return 24.620001;
                                    } else {
                                        if (f[24] <= -1.030354) {
                                            return 25.264999;
                                        } else {
                                            return 25.070001;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f[43] <= -0.137912) {
                                return 26.639999;
                            } else {
                                return 26.840000;
                            }
                        }
                    }
                }
            } else {
                if (f[1] <= 42.940805) {
                    if (f[5] <= -2.486452) {
                        if (f[21] <= -2.931730) {
                            if (f[0] <= -13.978854) {
                                if (f[31] <= 0.338812) {
                                    if (f[23] <= 0.997638) {
                                        if (f[3] <= -13.130296) {
                                            return 25.270000;
                                        } else {
                                            return 25.400000;
                                        }
                                    } else {
                                        return 25.809999;
                                    }
                                } else {
                                    if (f[49] <= 0.149614) {
                                        return 24.209999;
                                    } else {
                                        return 24.700001;
                                    }
                                }
                            } else {
                                if (f[10] <= 1.239563) {
                                    return 28.100000;
                                } else {
                                    if (f[46] <= -0.348747) {
                                        return 26.530001;
                                    } else {
                                        return 26.650000;
                                    }
                                }
                            }
                        } else {
                            if (f[4] <= 7.696340) {
                                if (f[0] <= -3.577345) {
                                    if (f[2] <= -28.557375) {
                                        if (f[46] <= -0.220219) {
                                            return 27.860000;
                                        } else {
                                            return 26.832857;
                                        }
                                    } else {
                                        if (f[36] <= 0.530397) {
                                            return 28.198572;
                                        } else {
                                            return 29.576667;
                                        }
                                    }
                                } else {
                                    if (f[35] <= 0.239968) {
                                        if (f[9] <= -2.730566) {
                                            return 28.433333;
                                        } else {
                                            return 28.929999;
                                        }
                                    } else {
                                        if (f[27] <= -0.062750) {
                                            return 29.322500;
                                        } else {
                                            return 29.910000;
                                        }
                                    }
                                }
                            } else {
                                if (f[0] <= -10.135229) {
                                    if (f[29] <= -1.532118) {
                                        if (f[38] <= 0.648211) {
                                            return 26.305000;
                                        } else {
                                            return 26.514999;
                                        }
                                    } else {
                                        if (f[39] <= 0.686225) {
                                            return 25.940000;
                                        } else {
                                            return 25.360001;
                                        }
                                    }
                                } else {
                                    if (f[28] <= 0.316947) {
                                        if (f[10] <= 0.653021) {
                                            return 28.193334;
                                        } else {
                                            return 27.560000;
                                        }
                                    } else {
                                        if (f[19] <= -1.547109) {
                                            return 26.540001;
                                        } else {
                                            return 26.960000;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (f[41] <= -0.364702) {
                            if (f[26] <= -0.270750) {
                                if (f[6] <= -5.763235) {
                                    if (f[36] <= -1.413704) {
                                        if (f[16] <= 0.660544) {
                                            return 24.600000;
                                        } else {
                                            return 24.629999;
                                        }
                                    } else {
                                        return 24.540001;
                                    }
                                } else {
                                    if (f[12] <= 2.643518) {
                                        if (f[31] <= -1.472906) {
                                            return 25.080000;
                                        } else {
                                            return 25.005000;
                                        }
                                    } else {
                                        return 25.270000;
                                    }
                                }
                            } else {
                                if (f[29] <= 1.589848) {
                                    if (f[28] <= -1.275701) {
                                        return 25.590000;
                                    } else {
                                        if (f[24] <= -0.088762) {
                                            return 25.905000;
                                        } else {
                                            return 25.809999;
                                        }
                                    }
                                } else {
                                    return 26.500000;
                                }
                            }
                        } else {
                            if (f[49] <= -0.116260) {
                                if (f[43] <= 0.213914) {
                                    if (f[28] <= 1.063499) {
                                        if (f[1] <= -14.386434) {
                                            return 27.510000;
                                        } else {
                                            return 27.570000;
                                        }
                                    } else {
                                        return 27.389999;
                                    }
                                } else {
                                    if (f[8] <= -7.757161) {
                                        return 26.850000;
                                    } else {
                                        if (f[27] <= -1.925425) {
                                            return 26.400000;
                                        } else {
                                            return 26.559999;
                                        }
                                    }
                                }
                            } else {
                                if (f[40] <= -0.429436) {
                                    if (f[24] <= 0.247854) {
                                        if (f[7] <= 12.482178) {
                                            return 26.370001;
                                        } else {
                                            return 26.340000;
                                        }
                                    } else {
                                        return 26.750000;
                                    }
                                } else {
                                    if (f[31] <= 0.664181) {
                                        if (f[16] <= 1.487737) {
                                            return 25.620001;
                                        } else {
                                            return 25.770000;
                                        }
                                    } else {
                                        return 25.139999;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f[8] <= -1.311880) {
                        if (f[0] <= 4.630192) {
                            if (f[3] <= 5.317350) {
                                if (f[1] <= 90.084213) {
                                    if (f[3] <= -15.727536) {
                                        return 29.879999;
                                    } else {
                                        return 29.770000;
                                    }
                                } else {
                                    return 29.299999;
                                }
                            } else {
                                if (f[42] <= 0.266469) {
                                    if (f[6] <= 13.918261) {
                                        if (f[34] <= -0.117631) {
                                            return 30.030001;
                                        } else {
                                            return 29.955000;
                                        }
                                    } else {
                                        return 30.420000;
                                    }
                                } else {
                                    if (f[1] <= 43.239294) {
                                        return 30.549999;
                                    } else {
                                        return 30.570000;
                                    }
                                }
                            }
                        } else {
                            if (f[28] <= -0.166999) {
                                return 31.190001;
                            } else {
                                if (f[4] <= -6.587126) {
                                    return 31.250000;
                                } else {
                                    return 31.270000;
                                }
                            }
                        }
                    } else {
                        if (f[23] <= -0.593382) {
                            if (f[16] <= -2.119440) {
                                return 26.620001;
                            } else {
                                if (f[11] <= -0.933130) {
                                    if (f[28] <= 0.158822) {
                                        return 28.309999;
                                    } else {
                                        return 28.200001;
                                    }
                                } else {
                                    return 28.090000;
                                }
                            }
                        } else {
                            if (f[8] <= 1.093004) {
                                if (f[37] <= 0.833640) {
                                    if (f[4] <= -20.058533) {
                                        return 29.719999;
                                    } else {
                                        if (f[9] <= -3.480063) {
                                            return 29.220001;
                                        } else {
                                            return 29.364000;
                                        }
                                    }
                                } else {
                                    return 28.799999;
                                }
                            } else {
                                if (f[3] <= 30.316818) {
                                    if (f[4] <= 0.959541) {
                                        if (f[14] <= -1.389047) {
                                            return 28.740000;
                                        } else {
                                            return 28.730000;
                                        }
                                    } else {
                                        if (f[21] <= 2.204452) {
                                            return 28.629999;
                                        } else {
                                            return 28.610001;
                                        }
                                    }
                                } else {
                                    return 29.250000;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (f[0] <= -36.159353) {
                if (f[2] <= 40.835482) {
                    if (f[3] <= -7.118173) {
                        if (f[1] <= 30.462994) {
                            if (f[2] <= 10.246777) {
                                if (f[19] <= -1.452468) {
                                    if (f[2] <= -7.020160) {
                                        if (f[4] <= -3.810184) {
                                            return 25.843334;
                                        } else {
                                            return 26.035000;
                                        }
                                    } else {
                                        if (f[47] <= -0.396938) {
                                            return 26.250000;
                                        } else {
                                            return 26.639999;
                                        }
                                    }
                                } else {
                                    if (f[19] <= 3.303843) {
                                        if (f[0] <= -62.137741) {
                                            return 26.193334;
                                        } else {
                                            return 26.871250;
                                        }
                                    } else {
                                        if (f[14] <= 1.284096) {
                                            return 27.405000;
                                        } else {
                                            return 27.840000;
                                        }
                                    }
                                }
                            } else {
                                if (f[5] <= -2.344317) {
                                    if (f[6] <= -19.070874) {
                                        if (f[36] <= 0.178386) {
                                            return 28.480000;
                                        } else {
                                            return 28.459999;
                                        }
                                    } else {
                                        return 28.540001;
                                    }
                                } else {
                                    if (f[11] <= -10.036125) {
                                        if (f[20] <= 1.627922) {
                                            return 27.129999;
                                        } else {
                                            return 27.049999;
                                        }
                                    } else {
                                        if (f[24] <= 1.683197) {
                                            return 27.820000;
                                        } else {
                                            return 27.360001;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f[19] <= -2.464847) {
                                return 28.020000;
                            } else {
                                if (f[27] <= -0.863649) {
                                    if (f[44] <= 0.004189) {
                                        return 28.820000;
                                    } else {
                                        return 29.000000;
                                    }
                                } else {
                                    return 29.680000;
                                }
                            }
                        }
                    } else {
                        if (f[0] <= -63.071690) {
                            if (f[31] <= 0.647446) {
                                if (f[23] <= -1.039708) {
                                    if (f[18] <= -3.457278) {
                                        return 27.639999;
                                    } else {
                                        if (f[13] <= -0.875392) {
                                            return 28.330000;
                                        } else {
                                            return 28.219999;
                                        }
                                    }
                                } else {
                                    if (f[9] <= -0.281860) {
                                        if (f[48] <= 0.351042) {
                                            return 27.379999;
                                        } else {
                                            return 27.270000;
                                        }
                                    } else {
                                        if (f[42] <= -0.171895) {
                                            return 27.500000;
                                        } else {
                                            return 27.530001;
                                        }
                                    }
                                }
                            } else {
                                if (f[30] <= -1.063300) {
                                    return 26.639999;
                                } else {
                                    return 25.950001;
                                }
                            }
                        } else {
                            if (f[16] <= -1.641483) {
                                if (f[44] <= -0.156493) {
                                    if (f[10] <= -9.664844) {
                                        if (f[15] <= 1.523386) {
                                            return 28.964999;
                                        } else {
                                            return 28.549999;
                                        }
                                    } else {
                                        if (f[40] <= 0.583879) {
                                            return 29.606667;
                                        } else {
                                            return 30.179999;
                                        }
                                    }
                                } else {
                                    if (f[22] <= 0.466961) {
                                        if (f[44] <= 0.032534) {
                                            return 28.273334;
                                        } else {
                                            return 27.726667;
                                        }
                                    } else {
                                        if (f[11] <= 2.687670) {
                                            return 28.549999;
                                        } else {
                                            return 29.070000;
                                        }
                                    }
                                }
                            } else {
                                if (f[49] <= -0.300092) {
                                    if (f[48] <= -0.484637) {
                                        if (f[30] <= 1.020740) {
                                            return 31.115001;
                                        } else {
                                            return 30.930000;
                                        }
                                    } else {
                                        if (f[32] <= 0.819500) {
                                            return 30.230000;
                                        } else {
                                            return 30.420000;
                                        }
                                    }
                                } else {
                                    if (f[36] <= -0.498103) {
                                        if (f[30] <= 0.915902) {
                                            return 30.590001;
                                        } else {
                                            return 29.970000;
                                        }
                                    } else {
                                        if (f[46] <= 1.707031) {
                                            return 29.398889;
                                        } else {
                                            return 27.879999;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f[5] <= -4.336390) {
                        if (f[18] <= 1.136822) {
                            if (f[45] <= 0.947326) {
                                if (f[16] <= -0.928072) {
                                    return 31.680000;
                                } else {
                                    if (f[35] <= 0.504011) {
                                        if (f[15] <= 0.201646) {
                                            return 32.150002;
                                        } else {
                                            return 32.355000;
                                        }
                                    } else {
                                        if (f[36] <= -0.904959) {
                                            return 32.459999;
                                        } else {
                                            return 32.650002;
                                        }
                                    }
                                }
                            } else {
                                return 34.580002;
                            }
                        } else {
                            if (f[46] <= 0.544119) {
                                if (f[7] <= 0.757547) {
                                    if (f[11] <= -0.690133) {
                                        return 30.540001;
                                    } else {
                                        return 30.510000;
                                    }
                                } else {
                                    return 30.600000;
                                }
                            } else {
                                return 30.990000;
                            }
                        }
                    } else {
                        if (f[7] <= -9.303683) {
                            if (f[37] <= 0.415676) {
                                return 31.920000;
                            } else {
                                return 32.759998;
                            }
                        } else {
                            if (f[2] <= 50.657993) {
                                if (f[0] <= -52.191605) {
                                    if (f[7] <= -6.582509) {
                                        return 28.170000;
                                    } else {
                                        if (f[22] <= 3.588161) {
                                            return 28.785000;
                                        } else {
                                            return 29.330000;
                                        }
                                    }
                                } else {
                                    if (f[5] <= 0.192060) {
                                        return 29.580000;
                                    } else {
                                        if (f[15] <= -4.642175) {
                                            return 29.490000;
                                        } else {
                                            return 29.415000;
                                        }
                                    }
                                }
                            } else {
                                if (f[44] <= -0.378201) {
                                    if (f[16] <= 2.454641) {
                                        return 30.410000;
                                    } else {
                                        return 30.299999;
                                    }
                                } else {
                                    if (f[37] <= -0.321369) {
                                        if (f[35] <= -0.021133) {
                                            return 29.820000;
                                        } else {
                                            return 29.850000;
                                        }
                                    } else {
                                        if (f[17] <= -0.117072) {
                                            return 29.933333;
                                        } else {
                                            return 30.020000;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (f[2] <= 18.376698) {
                    if (f[3] <= 17.160207) {
                        if (f[5] <= 32.469362) {
                            if (f[1] <= 41.272461) {
                                if (f[0] <= -11.937032) {
                                    if (f[3] <= -23.029173) {
                                        if (f[29] <= 0.483069) {
                                            return 27.814000;
                                        } else {
                                            return 28.660000;
                                        }
                                    } else {
                                        if (f[5] <= -14.375366) {
                                            return 30.745000;
                                        } else {
                                            return 29.208182;
                                        }
                                    }
                                } else {
                                    if (f[3] <= -20.933276) {
                                        if (f[19] <= -0.361424) {
                                            return 28.930000;
                                        } else {
                                            return 29.833846;
                                        }
                                    } else {
                                        if (f[15] <= 0.100419) {
                                            return 30.595789;
                                        } else {
                                            return 31.885556;
                                        }
                                    }
                                }
                            } else {
                                if (f[3] <= -3.556978) {
                                    if (f[6] <= -2.559846) {
                                        if (f[28] <= -1.384364) {
                                            return 32.145000;
                                        } else {
                                            return 31.510000;
                                        }
                                    } else {
                                        if (f[41] <= -0.160742) {
                                            return 30.466667;
                                        } else {
                                            return 31.033750;
                                        }
                                    }
                                } else {
                                    if (f[17] <= 2.200730) {
                                        if (f[26] <= 0.138667) {
                                            return 33.285000;
                                        } else {
                                            return 32.615000;
                                        }
                                    } else {
                                        return 31.540001;
                                    }
                                }
                            }
                        } else {
                            if (f[7] <= 0.185148) {
                                if (f[10] <= 0.982870) {
                                    if (f[33] <= -0.023829) {
                                        if (f[4] <= 12.664594) {
                                            return 27.135000;
                                        } else {
                                            return 27.430000;
                                        }
                                    } else {
                                        if (f[46] <= 0.163211) {
                                            return 27.709999;
                                        } else {
                                            return 27.785000;
                                        }
                                    }
                                } else {
                                    if (f[43] <= -0.544325) {
                                        if (f[47] <= 0.302009) {
                                            return 27.889999;
                                        } else {
                                            return 27.969999;
                                        }
                                    } else {
                                        if (f[16] <= 0.287956) {
                                            return 28.475000;
                                        } else {
                                            return 28.320000;
                                        }
                                    }
                                }
                            } else {
                                if (f[6] <= 7.528815) {
                                    if (f[1] <= 3.175256) {
                                        if (f[39] <= 0.447983) {
                                            return 29.540000;
                                        } else {
                                            return 29.040001;
                                        }
                                    } else {
                                        return 30.559999;
                                    }
                                } else {
                                    if (f[7] <= 14.994889) {
                                        return 28.160000;
                                    } else {
                                        if (f[46] <= -0.021624) {
                                            return 28.709999;
                                        } else {
                                            return 28.680000;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (f[5] <= -14.022857) {
                            if (f[24] <= -0.749335) {
                                if (f[18] <= 2.006377) {
                                    return 33.720001;
                                } else {
                                    if (f[35] <= -0.514526) {
                                        return 35.110001;
                                    } else {
                                        if (f[27] <= 0.073006) {
                                            return 34.470001;
                                        } else {
                                            return 34.394999;
                                        }
                                    }
                                }
                            } else {
                                if (f[37] <= -0.075045) {
                                    if (f[29] <= 1.779054) {
                                        return 33.630001;
                                    } else {
                                        return 33.509998;
                                    }
                                } else {
                                    return 32.869999;
                                }
                            }
                        } else {
                            if (f[8] <= 2.536614) {
                                if (f[26] <= -0.364664) {
                                    if (f[19] <= 0.897908) {
                                        if (f[30] <= 1.000616) {
                                            return 30.680000;
                                        } else {
                                            return 31.512000;
                                        }
                                    } else {
                                        if (f[47] <= -0.720244) {
                                            return 30.090000;
                                        } else {
                                            return 29.545000;
                                        }
                                    }
                                } else {
                                    if (f[5] <= -6.761951) {
                                        if (f[30] <= -0.323329) {
                                            return 32.997499;
                                        } else {
                                            return 32.080000;
                                        }
                                    } else {
                                        if (f[43] <= 0.146385) {
                                            return 31.304000;
                                        } else {
                                            return 31.836667;
                                        }
                                    }
                                }
                            } else {
                                if (f[37] <= 0.758914) {
                                    if (f[38] <= 0.369243) {
                                        if (f[45] <= 0.132484) {
                                            return 33.766666;
                                        } else {
                                            return 33.470000;
                                        }
                                    } else {
                                        if (f[45] <= 0.299972) {
                                            return 32.626667;
                                        } else {
                                            return 32.970001;
                                        }
                                    }
                                } else {
                                    if (f[23] <= -0.802105) {
                                        return 32.580002;
                                    } else {
                                        if (f[6] <= -3.920212) {
                                            return 31.719999;
                                        } else {
                                            return 31.130000;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f[5] <= -10.284215) {
                        if (f[4] <= -9.305056) {
                            if (f[0] <= 1.894413) {
                                if (f[30] <= -0.403017) {
                                    if (f[34] <= -0.892096) {
                                        return 34.700001;
                                    } else {
                                        if (f[29] <= -0.094575) {
                                            return 35.459999;
                                        } else {
                                            return 35.804998;
                                        }
                                    }
                                } else {
                                    if (f[43] <= -0.839562) {
                                        if (f[12] <= 0.091281) {
                                            return 34.279999;
                                        } else {
                                            return 33.700001;
                                        }
                                    } else {
                                        if (f[11] <= -3.360559) {
                                            return 34.590000;
                                        } else {
                                            return 34.869999;
                                        }
                                    }
                                }
                            } else {
                                if (f[14] <= -0.501600) {
                                    return 37.169998;
                                } else {
                                    return 36.520000;
                                }
                            }
                        } else {
                            if (f[21] <= -0.677087) {
                                if (f[37] <= 0.061778) {
                                    if (f[16] <= 2.134573) {
                                        return 31.850000;
                                    } else {
                                        return 31.790001;
                                    }
                                } else {
                                    if (f[4] <= 0.801466) {
                                        if (f[5] <= -12.894125) {
                                            return 33.060001;
                                        } else {
                                            return 33.029999;
                                        }
                                    } else {
                                        return 32.950001;
                                    }
                                }
                            } else {
                                if (f[44] <= -0.269828) {
                                    if (f[22] <= -3.711203) {
                                        return 34.250000;
                                    } else {
                                        return 33.700001;
                                    }
                                } else {
                                    if (f[0] <= -0.882059) {
                                        return 34.889999;
                                    } else {
                                        return 35.380001;
                                    }
                                }
                            }
                        }
                    } else {
                        if (f[1] <= -4.383171) {
                            if (f[39] <= 0.705003) {
                                if (f[0] <= -24.118896) {
                                    if (f[42] <= -0.412317) {
                                        if (f[17] <= -0.861593) {
                                            return 31.285000;
                                        } else {
                                            return 31.760000;
                                        }
                                    } else {
                                        if (f[7] <= -7.122923) {
                                            return 31.299999;
                                        } else {
                                            return 30.670000;
                                        }
                                    }
                                } else {
                                    if (f[23] <= 0.173776) {
                                        if (f[23] <= -1.692031) {
                                            return 31.738000;
                                        } else {
                                            return 32.543750;
                                        }
                                    } else {
                                        if (f[22] <= -0.988762) {
                                            return 29.980000;
                                        } else {
                                            return 31.322500;
                                        }
                                    }
                                }
                            } else {
                                if (f[1] <= -21.621511) {
                                    if (f[47] <= 0.383954) {
                                        return 32.450001;
                                    } else {
                                        return 32.950001;
                                    }
                                } else {
                                    if (f[12] <= -1.323556) {
                                        return 35.380001;
                                    } else {
                                        if (f[26] <= -0.098124) {
                                            return 33.340000;
                                        } else {
                                            return 34.330000;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f[2] <= 29.340228) {
                                if (f[25] <= 0.006153) {
                                    if (f[23] <= -2.139985) {
                                        return 32.000000;
                                    } else {
                                        if (f[17] <= 1.870425) {
                                            return 32.605001;
                                        } else {
                                            return 32.740002;
                                        }
                                    }
                                } else {
                                    return 33.320000;
                                }
                            } else {
                                if (f[16] <= 3.753745) {
                                    if (f[14] <= -1.278618) {
                                        if (f[49] <= -0.200826) {
                                            return 34.700001;
                                        } else {
                                            return 34.805000;
                                        }
                                    } else {
                                        if (f[49] <= 0.318349) {
                                            return 34.159999;
                                        } else {
                                            return 33.934999;
                                        }
                                    }
                                } else {
                                    if (f[3] <= -3.820499) {
                                        return 32.259998;
                                    } else {
                                        return 33.340000;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (f[2] <= -11.897192) {
            if (f[0] <= 64.490402) {
                if (f[5] <= -7.146995) {
                    if (f[1] <= 26.610287) {
                        if (f[20] <= -1.588264) {
                            if (f[47] <= 0.742090) {
                                return 32.770000;
                            } else {
                                if (f[22] <= 0.136401) {
                                    return 32.110001;
                                } else {
                                    return 32.119999;
                                }
                            }
                        } else {
                            if (f[28] <= -0.935174) {
                                if (f[43] <= 0.511496) {
                                    if (f[16] <= -1.101450) {
                                        return 31.000000;
                                    } else {
                                        if (f[21] <= 1.566187) {
                                            return 30.910000;
                                        } else {
                                            return 30.940001;
                                        }
                                    }
                                } else {
                                    return 31.770000;
                                }
                            } else {
                                if (f[34] <= -0.203952) {
                                    if (f[4] <= 18.286308) {
                                        return 30.360001;
                                    } else {
                                        if (f[25] <= -0.703804) {
                                            return 30.549999;
                                        } else {
                                            return 30.520000;
                                        }
                                    }
                                } else {
                                    if (f[7] <= -4.520395) {
                                        return 30.270000;
                                    } else {
                                        if (f[10] <= -3.129081) {
                                            return 29.900000;
                                        } else {
                                            return 29.700000;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (f[21] <= -1.129798) {
                            if (f[38] <= -0.057740) {
                                if (f[1] <= 55.996925) {
                                    if (f[27] <= 2.211190) {
                                        if (f[16] <= 0.186727) {
                                            return 31.660000;
                                        } else {
                                            return 31.760000;
                                        }
                                    } else {
                                        return 31.860001;
                                    }
                                } else {
                                    if (f[14] <= -1.224361) {
                                        if (f[11] <= -0.209410) {
                                            return 31.129999;
                                        } else {
                                            return 31.320000;
                                        }
                                    } else {
                                        return 30.670000;
                                    }
                                }
                            } else {
                                if (f[26] <= 0.763416) {
                                    if (f[6] <= 11.269754) {
                                        return 32.459999;
                                    } else {
                                        return 32.310001;
                                    }
                                } else {
                                    return 32.849998;
                                }
                            }
                        } else {
                            if (f[29] <= -1.872514) {
                                if (f[26] <= 0.431240) {
                                    return 31.730000;
                                } else {
                                    return 32.209999;
                                }
                            } else {
                                if (f[49] <= -0.254453) {
                                    if (f[9] <= -0.630258) {
                                        if (f[20] <= 0.158814) {
                                            return 32.439999;
                                        } else {
                                            return 32.430000;
                                        }
                                    } else {
                                        return 32.349998;
                                    }
                                } else {
                                    if (f[12] <= -2.430114) {
                                        if (f[3] <= 16.859838) {
                                            return 32.834999;
                                        } else {
                                            return 33.010000;
                                        }
                                    } else {
                                        if (f[49] <= 0.090918) {
                                            return 33.590000;
                                        } else {
                                            return 33.375000;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f[19] <= -0.166629) {
                        if (f[38] <= 0.746816) {
                            if (f[0] <= 31.462548) {
                                if (f[48] <= -0.455248) {
                                    if (f[16] <= 2.790751) {
                                        if (f[9] <= -0.891467) {
                                            return 27.340000;
                                        } else {
                                            return 27.309999;
                                        }
                                    } else {
                                        if (f[44] <= -0.432758) {
                                            return 26.840000;
                                        } else {
                                            return 26.870001;
                                        }
                                    }
                                } else {
                                    if (f[22] <= -2.125961) {
                                        return 28.780001;
                                    } else {
                                        if (f[49] <= -0.528254) {
                                            return 28.170000;
                                        } else {
                                            return 27.776667;
                                        }
                                    }
                                }
                            } else {
                                if (f[16] <= -1.682532) {
                                    return 28.700001;
                                } else {
                                    if (f[19] <= -1.355079) {
                                        return 29.309999;
                                    } else {
                                        return 29.379999;
                                    }
                                }
                            }
                        } else {
                            if (f[46] <= -0.326379) {
                                return 31.790001;
                            } else {
                                if (f[3] <= -2.004748) {
                                    if (f[0] <= 50.289845) {
                                        return 29.370001;
                                    } else {
                                        return 29.280001;
                                    }
                                } else {
                                    if (f[20] <= -0.145688) {
                                        return 30.020000;
                                    } else {
                                        return 30.209999;
                                    }
                                }
                            }
                        }
                    } else {
                        if (f[11] <= -2.012904) {
                            if (f[10] <= -0.685403) {
                                return 32.950001;
                            } else {
                                return 32.820000;
                            }
                        } else {
                            if (f[7] <= -7.158556) {
                                if (f[17] <= -2.985856) {
                                    return 30.969999;
                                } else {
                                    if (f[45] <= -0.111197) {
                                        if (f[47] <= 0.272418) {
                                            return 30.505000;
                                        } else {
                                            return 30.580000;
                                        }
                                    } else {
                                        if (f[32] <= 0.358383) {
                                            return 30.400000;
                                        } else {
                                            return 30.370001;
                                        }
                                    }
                                }
                            } else {
                                if (f[41] <= -0.626680) {
                                    return 29.410000;
                                } else {
                                    if (f[14] <= 0.664386) {
                                        return 29.690001;
                                    } else {
                                        if (f[39] <= -0.480677) {
                                            return 29.820000;
                                        } else {
                                            return 29.910000;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (f[1] <= -5.207374) {
                    if (f[10] <= 2.172902) {
                        if (f[17] <= -1.150473) {
                            return 33.099998;
                        } else {
                            if (f[13] <= -4.148150) {
                                return 32.700001;
                            } else {
                                if (f[36] <= -0.207113) {
                                    return 32.560001;
                                } else {
                                    return 32.590000;
                                }
                            }
                        }
                    } else {
                        if (f[32] <= 0.013465) {
                            return 32.230000;
                        } else {
                            return 31.709999;
                        }
                    }
                } else {
                    if (f[17] <= -2.418582) {
                        return 33.669998;
                    } else {
                        if (f[14] <= 0.039075) {
                            if (f[25] <= 0.888949) {
                                if (f[8] <= 4.225615) {
                                    if (f[42] <= -0.017850) {
                                        return 35.139999;
                                    } else {
                                        if (f[34] <= -0.242666) {
                                            return 35.220001;
                                        } else {
                                            return 35.200001;
                                        }
                                    }
                                } else {
                                    return 35.360001;
                                }
                            } else {
                                if (f[8] <= -5.224154) {
                                    return 34.790001;
                                } else {
                                    return 34.680000;
                                }
                            }
                        } else {
                            if (f[48] <= 0.220427) {
                                return 35.900002;
                            } else {
                                return 35.669998;
                            }
                        }
                    }
                }
            }
        } else {
            if (f[0] <= 65.748768) {
                if (f[5] <= 3.125645) {
                    if (f[2] <= 29.407123) {
                        if (f[3] <= 4.871348) {
                            if (f[1] <= 27.619146) {
                                if (f[3] <= -17.401879) {
                                    if (f[4] <= 1.002840) {
                                        if (f[28] <= 0.918339) {
                                            return 31.713333;
                                        } else {
                                            return 31.340000;
                                        }
                                    } else {
                                        if (f[11] <= 2.819791) {
                                            return 32.440000;
                                        } else {
                                            return 33.160000;
                                        }
                                    }
                                } else {
                                    if (f[9] <= -2.740120) {
                                        if (f[27] <= -0.099832) {
                                            return 32.153333;
                                        } else {
                                            return 32.793332;
                                        }
                                    } else {
                                        if (f[5] <= -13.916511) {
                                            return 34.260000;
                                        } else {
                                            return 33.335334;
                                        }
                                    }
                                }
                            } else {
                                if (f[5] <= -16.752219) {
                                    if (f[33] <= -0.399101) {
                                        if (f[26] <= -1.827713) {
                                            return 32.509998;
                                        } else {
                                            return 33.010000;
                                        }
                                    } else {
                                        if (f[33] <= -0.269453) {
                                            return 34.020000;
                                        } else {
                                            return 33.584999;
                                        }
                                    }
                                } else {
                                    if (f[5] <= -13.650036) {
                                        if (f[39] <= 0.353040) {
                                            return 34.650001;
                                        } else {
                                            return 34.050001;
                                        }
                                    } else {
                                        if (f[47] <= -0.037722) {
                                            return 34.923333;
                                        } else {
                                            return 36.049999;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f[17] <= 1.518416) {
                                if (f[37] <= -0.517140) {
                                    if (f[8] <= 12.545321) {
                                        if (f[2] <= 8.411496) {
                                            return 34.565001;
                                        } else {
                                            return 35.413334;
                                        }
                                    } else {
                                        if (f[21] <= 0.853032) {
                                            return 36.869999;
                                        } else {
                                            return 35.980000;
                                        }
                                    }
                                } else {
                                    if (f[6] <= -13.834239) {
                                        if (f[19] <= 0.066242) {
                                            return 36.730000;
                                        } else {
                                            return 37.596667;
                                        }
                                    } else {
                                        if (f[29] <= -0.807059) {
                                            return 35.650002;
                                        } else {
                                            return 36.180000;
                                        }
                                    }
                                }
                            } else {
                                if (f[1] <= -36.719751) {
                                    if (f[40] <= 0.367032) {
                                        if (f[46] <= 0.802570) {
                                            return 34.193333;
                                        } else {
                                            return 33.450001;
                                        }
                                    } else {
                                        if (f[27] <= -0.633251) {
                                            return 34.480000;
                                        } else {
                                            return 34.885000;
                                        }
                                    }
                                } else {
                                    if (f[43] <= 0.126976) {
                                        return 35.270000;
                                    } else {
                                        if (f[36] <= 0.178830) {
                                            return 35.599998;
                                        } else {
                                            return 35.520000;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (f[1] <= 29.653897) {
                            if (f[29] <= -0.520806) {
                                if (f[49] <= 0.111574) {
                                    if (f[17] <= -1.971624) {
                                        return 34.660000;
                                    } else {
                                        if (f[28] <= -0.175569) {
                                            return 35.330002;
                                        } else {
                                            return 35.270000;
                                        }
                                    }
                                } else {
                                    if (f[48] <= -0.490520) {
                                        return 36.599998;
                                    } else {
                                        if (f[37] <= -0.437459) {
                                            return 35.849998;
                                        } else {
                                            return 35.860001;
                                        }
                                    }
                                }
                            } else {
                                if (f[27] <= -2.634665) {
                                    return 38.090000;
                                } else {
                                    if (f[42] <= 0.105089) {
                                        if (f[0] <= 47.790171) {
                                            return 36.074000;
                                        } else {
                                            return 36.504999;
                                        }
                                    } else {
                                        if (f[17] <= -2.080352) {
                                            return 36.700001;
                                        } else {
                                            return 36.949999;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f[47] <= -0.652655) {
                                if (f[6] <= -2.497181) {
                                    if (f[6] <= -6.172401) {
                                        return 38.389999;
                                    } else {
                                        return 38.290001;
                                    }
                                } else {
                                    return 38.930000;
                                }
                            } else {
                                if (f[22] <= 0.540881) {
                                    if (f[30] <= -0.159165) {
                                        return 37.240002;
                                    } else {
                                        if (f[20] <= -0.809317) {
                                            return 37.130001;
                                        } else {
                                            return 37.180000;
                                        }
                                    }
                                } else {
                                    if (f[44] <= -0.956334) {
                                        return 37.709999;
                                    } else {
                                        if (f[11] <= 1.904713) {
                                            return 37.810001;
                                        } else {
                                            return 37.820000;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f[6] <= -1.956660) {
                        if (f[35] <= 0.095104) {
                            if (f[25] <= -0.624915) {
                                if (f[40] <= 0.534732) {
                                    return 35.910000;
                                } else {
                                    if (f[34] <= -0.182339) {
                                        if (f[9] <= -2.117324) {
                                            return 34.930000;
                                        } else {
                                            return 34.959999;
                                        }
                                    } else {
                                        return 35.020000;
                                    }
                                }
                            } else {
                                if (f[37] <= 0.818009) {
                                    return 32.810001;
                                } else {
                                    if (f[25] <= -0.016609) {
                                        return 34.270000;
                                    } else {
                                        if (f[1] <= 68.114532) {
                                            return 33.914999;
                                        } else {
                                            return 33.849998;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f[45] <= 0.411963) {
                                if (f[19] <= 0.717406) {
                                    if (f[16] <= -0.548739) {
                                        if (f[45] <= 0.038350) {
                                            return 31.860001;
                                        } else {
                                            return 32.145000;
                                        }
                                    } else {
                                        return 30.760000;
                                    }
                                } else {
                                    if (f[36] <= -0.062825) {
                                        return 33.840000;
                                    } else {
                                        return 32.730000;
                                    }
                                }
                            } else {
                                if (f[42] <= -0.108632) {
                                    return 33.130001;
                                } else {
                                    if (f[39] <= -0.178695) {
                                        return 34.450001;
                                    } else {
                                        if (f[12] <= 0.317567) {
                                            return 33.930000;
                                        } else {
                                            return 34.189999;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (f[48] <= 0.029594) {
                            if (f[43] <= -0.007418) {
                                if (f[10] <= -4.715579) {
                                    if (f[49] <= -0.007270) {
                                        return 31.860001;
                                    } else {
                                        if (f[18] <= 0.376075) {
                                            return 32.389999;
                                        } else {
                                            return 32.840000;
                                        }
                                    }
                                } else {
                                    if (f[3] <= 16.212938) {
                                        if (f[0] <= 31.212317) {
                                            return 30.626667;
                                        } else {
                                            return 30.480000;
                                        }
                                    } else {
                                        return 31.010000;
                                    }
                                }
                            } else {
                                if (f[7] <= -17.011789) {
                                    return 33.689999;
                                } else {
                                    if (f[35] <= 1.224508) {
                                        if (f[31] <= -0.246928) {
                                            return 33.419998;
                                        } else {
                                            return 33.350001;
                                        }
                                    } else {
                                        return 33.189999;
                                    }
                                }
                            }
                        } else {
                            if (f[0] <= 31.812754) {
                                if (f[36] <= 0.530462) {
                                    if (f[46] <= -0.406171) {
                                        if (f[24] <= -0.654887) {
                                            return 30.875000;
                                        } else {
                                            return 30.530001;
                                        }
                                    } else {
                                        if (f[23] <= 0.551742) {
                                            return 30.266667;
                                        } else {
                                            return 30.010000;
                                        }
                                    }
                                } else {
                                    if (f[32] <= -0.428742) {
                                        return 29.490000;
                                    } else {
                                        if (f[3] <= -25.937769) {
                                            return 29.240000;
                                        } else {
                                            return 29.209999;
                                        }
                                    }
                                }
                            } else {
                                if (f[4] <= 3.637143) {
                                    if (f[48] <= 0.274923) {
                                        if (f[29] <= -0.081593) {
                                            return 31.780001;
                                        } else {
                                            return 31.750000;
                                        }
                                    } else {
                                        return 32.040001;
                                    }
                                } else {
                                    if (f[42] <= 0.045365) {
                                        if (f[5] <= 30.623334) {
                                            return 32.580002;
                                        } else {
                                            return 32.549999;
                                        }
                                    } else {
                                        return 32.250000;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (f[23] <= -0.150120) {
                    if (f[2] <= 34.878212) {
                        if (f[1] <= 33.834394) {
                            if (f[42] <= 0.112760) {
                                if (f[5] <= -17.907674) {
                                    if (f[48] <= -0.287483) {
                                        return 39.220001;
                                    } else {
                                        return 39.070000;
                                    }
                                } else {
                                    if (f[1] <= 1.519752) {
                                        return 39.799999;
                                    } else {
                                        return 39.689999;
                                    }
                                }
                            } else {
                                if (f[48] <= -0.379759) {
                                    return 38.529999;
                                } else {
                                    if (f[24] <= -0.704427) {
                                        return 38.430000;
                                    } else {
                                        return 38.480000;
                                    }
                                }
                            }
                        } else {
                            if (f[13] <= -3.104928) {
                                return 36.520000;
                            } else {
                                if (f[32] <= 1.058052) {
                                    if (f[30] <= 0.303511) {
                                        return 37.840000;
                                    } else {
                                        return 37.630001;
                                    }
                                } else {
                                    if (f[8] <= -5.415648) {
                                        return 38.130001;
                                    } else {
                                        return 37.959999;
                                    }
                                }
                            }
                        }
                    } else {
                        if (f[12] <= -0.998176) {
                            if (f[38] <= -0.213728) {
                                return 40.060001;
                            } else {
                                return 40.639999;
                            }
                        } else {
                            return 41.279999;
                        }
                    }
                } else {
                    if (f[32] <= -0.831706) {
                        if (f[30] <= -0.560659) {
                            return 32.040001;
                        } else {
                            return 33.299999;
                        }
                    } else {
                        if (f[39] <= 0.084189) {
                            if (f[6] <= 4.010904) {
                                if (f[4] <= 3.145249) {
                                    if (f[2] <= 12.261317) {
                                        if (f[31] <= 0.799013) {
                                            return 35.673332;
                                        } else {
                                            return 35.919998;
                                        }
                                    } else {
                                        if (f[21] <= -1.470848) {
                                            return 36.349998;
                                        } else {
                                            return 36.139999;
                                        }
                                    }
                                } else {
                                    if (f[20] <= 0.049474) {
                                        if (f[48] <= 0.170093) {
                                            return 37.000000;
                                        } else {
                                            return 36.889999;
                                        }
                                    } else {
                                        return 37.320000;
                                    }
                                }
                            } else {
                                if (f[36] <= 0.190451) {
                                    if (f[23] <= 2.813606) {
                                        if (f[4] <= -22.411770) {
                                            return 35.380001;
                                        } else {
                                            return 35.405001;
                                        }
                                    } else {
                                        return 35.900002;
                                    }
                                } else {
                                    if (f[36] <= 0.555436) {
                                        if (f[21] <= 0.318655) {
                                            return 34.410000;
                                        } else {
                                            return 34.595001;
                                        }
                                    } else {
                                        return 34.880001;
                                    }
                                }
                            }
                        } else {
                            if (f[36] <= -0.078153) {
                                if (f[6] <= 1.583018) {
                                    if (f[16] <= 4.104724) {
                                        if (f[47] <= -0.074262) {
                                            return 39.240002;
                                        } else {
                                            return 39.119999;
                                        }
                                    } else {
                                        if (f[1] <= -33.433142) {
                                            return 38.700001;
                                        } else {
                                            return 38.740002;
                                        }
                                    }
                                } else {
                                    return 37.849998;
                                }
                            } else {
                                if (f[22] <= -0.447976) {
                                    if (f[25] <= -0.215699) {
                                        return 37.730000;
                                    } else {
                                        return 38.200001;
                                    }
                                } else {
                                    if (f[25] <= 0.083489) {
                                        if (f[41] <= 0.212684) {
                                            return 37.014999;
                                        } else {
                                            return 37.520000;
                                        }
                                    } else {
                                        if (f[22] <= 1.375495) {
                                            return 36.665001;
                                        } else {
                                            return 36.470001;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

