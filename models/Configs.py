class Configs():
    # AIDE settings
    feature_dim = [16, 32, 64]
    abs_ah_key = "accumulated_abs_ah"
    current_key = "current"
    diff_time_key = "diff_time"
    soc_key = "soc"
    v_min_key = "v_min"
    v_max_key = "v_cell"
    t_min_key = "t_min"
    t_max_key = "t_min"
    cap_nom_key = "cap_nominal"
    v_diff_key = 'diff_v'
    vmin_low_tp_key = "htp_vmin"
    vmin_high_tp_key = "htp_cell"

    all_keys = [abs_ah_key, diff_time_key,
                current_key, soc_key, v_min_key, v_max_key, t_min_key, cap_nom_key, v_diff_key]
    dim_in = len(all_keys)


    def __init__(self):
        self.dim_feature = [16, 32, 64]
        self.out_num = 2
        self.kernel_size = 9
        self.encoder_depth = 10
        self.decoder_depth = 10
        self.max_length = 10000
        self.class_num = 2
        self.loc_lim = 300
        self.move_rate = 0.001
        self.eps = 0.1
        self.affine = 1
        self.num_heads = 2
        self.mlp_ratio = 1

        # # original settings
        # abs_ah_key = "abs_ah"
        # current_key = "current"
        # diff_time_key = "diff_time"
        # soc_key = "soc"
        # v_min_key = "vmin"
        # v_max_key = "vmax"
        # t_min_key = "tmin"
        # t_max_key = "tmax"
        # cap_nom_key = "cap_nominal"
        # vmin_low_tp_key = "vmin_low_tp"
        # vmin_high_tp_key = "vmin_high_tp"

