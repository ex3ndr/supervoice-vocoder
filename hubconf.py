dependencies = ['torch', 'torchaudio']

def bigvsan(pretrained=True):
    from supervoice_vocoder.bigvsan import BigVSAN
    model = BigVSAN()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/bigvsan.pt")
        model.load_state_dict(checkpoint)
    return model
            