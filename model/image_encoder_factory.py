from model.openclip_image_encoder import OpenClipImageEncoder


def build_image_encoder(config, device):
    encoder_type = config.get("image_encoder_type", "openclip").lower()

    if encoder_type == "openclip":
        return OpenClipImageEncoder(
            model_name=config["image_encoder_model_name"],
            pretrained=config["image_encoder_pretrained"],
            device=device,
        )

    raise ValueError(f"Unsupported image_encoder_type: {config['image_encoder_type']}")
