def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    anchors = []
    
    stride = image_size / feature_size
    
    for i in range(feature_size):          # row (y)
        for j in range(feature_size):      # col (x)
            # Compute center
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            # For each scale and aspect ratio
            for s in scales:
                for r in aspect_ratios:
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([x1, y1, x2, y2])
    
    return anchors