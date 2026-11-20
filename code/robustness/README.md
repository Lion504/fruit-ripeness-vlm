# Banana robustness sets

Scripts that made the three banana robustness test sets from BananaImageBD
test photos (fixed seeds):

- small in the frame: the photo shrunk to 20-40% of the frame on a real-photo
  or generated background (`extra_sets.py`, `build_framing`);
- cut off: the banana shifted so about 30% of it leaves the frame;
- new backgrounds (`bg_new2`): the photo shrunk to 50-80% on crops of other
  real photos.

The finished images are not published: some backgrounds come from photos whose
licences do not allow redistribution. The test lists in `../../test_sets/`
name the source BananaImageBD photo of every robustness image. These scripts
need the banana masks and background folders described in `hardset.py` and
`extra_sets.py`, and `numpy`, `pandas`, `scikit-learn`, `pillow`.
