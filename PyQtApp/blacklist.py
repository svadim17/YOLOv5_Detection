import yaml


def load_config():
    with open('server_conf.yaml', encoding='utf-8') as f:
        config = dict(yaml.load(f, Loader=yaml.SafeLoader))
        return config


conf = load_config()
freqs = conf['freq_codes']
print(freqs)

for freq, value in freqs.items():
    val = bytes.fromhex(value)
    print(f'{freq} : {val}')
