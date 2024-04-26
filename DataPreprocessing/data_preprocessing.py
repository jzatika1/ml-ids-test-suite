def simplify_attack_names(y):
    def map_web_attack(label):
        if 'Web Attack' in label:
            return 'Web Attack'
        return label

    attack_map = {
        'DoS Hulk': 'Dos Attack',
        'DoS GoldenEye': 'Dos Attack',
        'DoS slowloris': 'Dos Attack',
        'DoS Slowhttptest': 'Dos Attack',
        'DDoS': 'Dos Attack',
        'ddos': 'Dos Attack',
        'dos': 'Dos Attack',
        'Web Attack': 'Web Attack',
        'xss': 'Web Attack',
        'injection': 'Web Attack',
        'PortScan': 'Scanning And Probing',
        'scanning': 'Scanning And Probing',
        'FTP-Patator': 'Brute Force Attack',
        'SSH-Patator': 'Brute Force Attack',
        'password': 'Brute Force Attack',
    }

    return y.map(map_web_attack).map(attack_map).fillna(y).str.title()