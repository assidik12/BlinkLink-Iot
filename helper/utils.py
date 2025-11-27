import cv2
import dlib
import helper.config as config


class Utils:
    """Utility helpers for CV + UI operations.

    Provides methods:
    - draw_text_center(surface, text, font, color, rect)
    - resize_frame_for_detection(frame, scale=None)
    - scale_rect(rect, scale)
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config

    def draw_text_center(self, surface, text, font, color, rect):
        """Render text centered inside a pygame rect on the given surface."""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

    def resize_frame_for_detection(self, frame, scale=None):
        """Resize frame for faster detection. If scale is None use config.DETECTION_SCALE."""
        if scale is None:
            scale = getattr(self.cfg, "DETECTION_SCALE", 0.5)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return small, scale

    def scale_rect(self, rect, scale):
        """Scale a dlib rectangle from scaled detection back to original frame coordinates."""
        return dlib.rectangle(
            int(rect.left() / scale),
            int(rect.top() / scale),
            int(rect.right() / scale),
            int(rect.bottom() / scale)
        )


# Singleton instance for convenience
util = Utils(config)

# Backward-compatible function wrappers
def draw_text_center(surface, text, font, color, rect):
    return util.draw_text_center(surface, text, font, color, rect)


def resize_frame_for_detection(frame, scale=None):
    return util.resize_frame_for_detection(frame, scale)


def scale_rect(rect, scale):
    return util.scale_rect(rect, scale)
