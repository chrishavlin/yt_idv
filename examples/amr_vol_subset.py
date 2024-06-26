import yt

import yt_idv

ds = yt.load_sample("IsolatedGalaxy")
c = ds.domain_center
le = c - ds.quan(.05, 'code_length')
re = c + ds.quan(.05, 'code_length')
r = ds.region(c, le, re)
rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(r, "density", no_ghost=True)
rc.run()
