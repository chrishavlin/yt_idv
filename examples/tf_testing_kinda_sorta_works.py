import yt
from yt_idv import render_context
from yt_idv.scene_data.block_collection import BlockCollection
from yt_idv.scene_components.blocks import BlockRendering
import numpy as np

ds = yt.load_sample('IsolatedGalaxy')
ad = ds.all_data()

rc = render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds, None)

# manually build the block collection, with scale=True
ad_block_data = BlockCollection(data_source=ad, scale=True)
ad_block_data.add_data("Density", no_ghost=True)
# now a rendering context with the block collection
ad_block_rend = BlockRendering(data=ad_block_data)


# each block gets a texture, data gets absoluted and normalized
#   n_data = np.abs(block.my_data[0]).copy(order="F").astype("float32").d
#   if self.max_val != self.min_val:
#         n_data = (n_data - self.min_val) / (
#             (self.max_val - self.min_val)
#           )
#
# ad[('gas', 'density')].max()
# Out[8]: unyt_quantity(5.92159918e-26, 'g/cm**3')
# ad[('gas', 'density')].min()
# Out[9]: unyt_quantity(3.33061222e-30, 'g/cm**3')
#

def block_normalized_value(raw_data_val, block_collection: BlockCollection):
    val_range = np.log10(block_collection.max_val) - np.log10(block_collection.min_val)
    return (np.log10(np.abs(raw_data_val)) - np.log10(block_collection.min_val)) / val_range


print([ad_block_data.min_val, ad_block_data.max_val])
print(np.log10([ad_block_data.min_val, ad_block_data.max_val]))

# the following are uniforms that get passed down

# ad_block_rend.tf_min = 0.0   # in data-normalized space
# ad_block_rend.tf_max = 1.0   # in data-normalized space
# ad_block_rend.tf_log = True  # apply in log-space


# ad_block_rend.render_method = 'max_intensity' # default
ad_block_rend.render_method = 'transfer_function'

# ad_block_rend.transfer_function.data

# Out[4]:
# array([[[255, 255, 255, 255]],
#        [[255, 255, 255, 255]],
#        [[255, 255, 255, 255]],
#        ...,
#        [[255, 255, 255, 255]],
#        [[255, 255, 255, 255]],
#        [[255, 255, 255, 255]]], dtype=uint8)

# ad_block_rend.transfer_function.data.shape
# Out[6]: (256, 1, 4)

# for data in (0, 1)
# ad_block_rend.transfer_function.data = (data * 255).astype("u1")

# R = ad_block_rend.transfer_function.data[:, 0, 0]
# G = ad_block_rend.transfer_function.data[:, 0, 1]
# B = ad_block_rend.transfer_function.data[:, 0, 2]
# a = ad_block_rend.transfer_function.data[:, 0, 3]

# try a gaussian

center = block_normalized_value(1e4, ad_block_data)
center = 1e4
print(f"new center{center}")
ad_block_rend.tf_log = False
#
# def maybelog(x):
#     # if ad_block_rend.tf_log:
#     #     return np.log(x)
#     return x

channel = np.linspace(0, 1, 256)

c1 = ds.arr(1e-28, 'g/cm**3').to('code_mass/code_length**3').value
c2 = ds.arr(1e-27, 'g/cm**3').to('code_mass/code_length**3').value
center1 = block_normalized_value(c1, ad_block_data);
center2 = block_normalized_value(c2, ad_block_data);

print((center1, center2))
# center1 = 0.001
# center2 = 0.005
# center3 = 0.01
R = np.exp(-(((channel) - (center1)) / .1) ** 2)
G = np.exp(-(((channel) - (center2)) / .1) ** 2)
B = np.zeros(channel.shape)
a = np.ones(channel.shape)

c3 = ds.arr(1e-24, 'g/cm**3').to('code_mass/code_length**3').value
center3 = block_normalized_value(c3, ad_block_data);
R = np.exp(-(((channel) - (center3)) / .1) ** 2)
G = np.exp(-(((channel) - (center3)) / .1) ** 2)
B = np.exp(-(((channel) - (center3)) / .1) ** 2)

tf_data = np.zeros(ad_block_rend.transfer_function.data.shape)
tf_data[:, 0, 0] = R
tf_data[:, 0, 1] = G
tf_data[:, 0, 2] = B
tf_data[:, 0, 3] = a

ad_block_rend.transfer_function.data = (tf_data * 255).astype("u1")

# add it to the scene
sg.data_objects.append(ad_block_data)
sg.components.append(ad_block_rend)

rc.run()
