import re


if __name__ == "__main__":
    regExToFind = r"^(EEG )?([A-Z]{1,2}[1-9]*)(-[a-z]?[1-9]*)?"
    channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
    print(len(channels))
    for i, channel in enumerate(channels):
        result = re.search(regExToFind, channel, flags=re.IGNORECASE)
        if result.group(2) is not None:
            electrode = result.group(2)
        else:
            electrode = channel
        channels[i] = "{}-{}".format(electrode, "REF")

    print(len(channels))
    print(channels)

