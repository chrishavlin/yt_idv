import yt

import yt_idv

ds = yt.load_sample("IsolatedGalaxy")

rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds, "density", no_ghost=True)
rc.run()




ds = yt.load('flash_idv_tests/m1.0_p16_b2.0_300k_plt50/multitidal_hdf5_plt_cnt_0200')
data_source = ds.all_data()
data_source.tiles.set_fields(["density"], [False], no_ghost=True)
for i, block in enumerate(data_source.tiles.traverse()):
    if i > 10:
        break
    print(i)
    print(block.RightEdge)
    print(block.LeftEdge)
