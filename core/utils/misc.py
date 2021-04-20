import logging
def get_root_logger(rank, filename=None, log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            filename=filename if rank == 0 else None,
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    if rank != 0:
        logger.setLevel('ERROR')
    return logger

def make_anchors(dataset, input_size=416):
    if dataset=='refeit':
        anchors = '30,36,  78,46,  48,86,  149,79,  82,148,  331,93,  156,207,  381,163,  329,285'
    elif dataset=='flickr':
        anchors = '29,26,  55,58,  137,71,  82,121,  124,205,  204,132,  209,263,  369,169,  352,294'
    else:
        anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = [float(x) for x in anchors.split(',')]
    anchors = [x * input_size / 416 for x in anchors]
    anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]
    return anchors_full
