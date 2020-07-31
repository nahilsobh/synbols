from icu import LocaleData
import logging
import numpy as np
from collections import defaultdict, Counter


def _filter_fonts(all_fonts, font_properties, font_cluters):
    if font_properties is None:
        white_list = set(all_fonts)
    else:
        white_list = {font for font in all_fonts if (font_properties[font]['empty_count'] == 0 and
                                                     font_properties[font]['no_bold_count'] == 0)}

    if font_cluters is not None:

        for cluster in font_cluters:

            white_cluster = white_list.intersection({e for e, _ in cluster})
            if len(white_cluster) > 0:
                kept_font = white_cluster.pop()
                logging.debug("Cluster of size %d, %d are in the white list, keeping %s", len(cluster),
                              len(white_cluster) + 1,
                              kept_font)
            else:
                logging.debug("Cluster of size %d, none are in the whitelist", len(cluster))

            # Remove the rest of the cluster from the whitelist
            white_list.difference_update(white_cluster)

    return list(white_list)


class Alphabet:
    """Combines fonts and symbols for a given language."""

    def __init__(self, name, fonts, symbols):
        self.name = name
        self.symbols = symbols
        self.all_fonts = fonts
        self.fonts = fonts
        self.font_properties = None
        self.font_clusters = None

    def filter_fonts(self, font_properties=None, font_clusters=None):
        self.font_properties = font_properties
        self.font_clusters = font_clusters

        self.fonts = _filter_fonts(self.all_fonts, font_properties, font_clusters)
        logging.debug("Filtering fonts for alphabet %s from %d to %d", self.name, len(self.all_fonts), len(self.fonts))


def get_char_set(language, add_cases=False):
    char_set = list(LocaleData(language).getExemplarSet())

    if not add_cases:
        return char_set

    n_char = len(char_set)

    char_set_alt = [char.swapcase() for char in char_set]
    char_set = set(char_set).union(char_set_alt)

    n_char_new = len(char_set)
    if n_char != n_char_new:
        logging.debug("inflating %s from %d to %d using uppercase.", language, n_char, n_char_new)

    return list(char_set)


def flatten_attr(attr, ctxt=None):
    flat_dict = {}
    if isinstance(attr, (list, tuple, np.ndarray)):
        for i, val in enumerate(attr):
            flat_dict.update(flatten_attr(val, ctxt + '[%d]' % i))

    elif isinstance(attr, dict):
        for key, val in attr.items():
            if ctxt is None:
                sub_ctxt = key
            else:
                sub_ctxt = ctxt + '.%s' % key
            flat_dict.update(flatten_attr(val, sub_ctxt))
    else:
        flat_dict = {ctxt: attr}
    return flat_dict


def _extract_axis(y, axis_name, max_val):
    if axis_name is None:
        return [None] * max_val

    counter = Counter([attr[axis_name] for attr in y])
    return [e for e, _ in counter.most_common(max_val)]


def make_img_grid(x, y, h_axis='char', v_axis='font', n_row=20, n_col=40):
    h_values = _extract_axis(y, h_axis, n_col)
    v_values = _extract_axis(y, v_axis, n_row)

    attr_map = defaultdict(list)
    for i, attr in enumerate(y):
        attr_map[(attr.get(h_axis), attr.get(v_axis))].append(i)

    img_grid = []
    blank_image = np.zeros(x.shape[1:], dtype=x.dtype)

    for v_value in v_values:
        img_row = []
        for h_value in h_values:
            if len(attr_map[(h_value, v_value)]) > 0:

                idx = attr_map[(h_value, v_value)].pop(0)
                img_row.append(x[idx])
            else:
                img_row.append(blank_image)

        img_grid.append(np.hstack(img_row))

    img_grid = np.vstack(img_grid)

    if img_grid.shape[-1] == 1:
        img_grid = img_grid.squeeze(axis=-1)

    return img_grid, h_values, v_values


SYMBOL_MAP = {
    'latin': get_char_set("en_US"),
    'telugu': get_char_set("te"),
    'thai': get_char_set("th"),
    'vietnamese': get_char_set("vi"),
    'arabic': get_char_set("ar"),
    'hebrew': get_char_set("iw_IL"),
    # 'khmer': get_char_set("km"),  # XXX: see note above
    'tamil': get_char_set("ta"),
    'gujarati': get_char_set("gu"),
    'bengali': get_char_set("bn"),
    'malayalam': get_char_set("ml"),
    'greek': get_char_set("el_GR"),
    'cyrillic': list(u"АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"),
    'korean': get_char_set("ko_KR"),
    'chinese-simplified': get_char_set("zh-CN")
}