import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Belirtilen isimle yapılandırılmış bir logger (kaydedici) döndürür.
    Konsola düzenli formatta çıktı verir.
    """
    logger = logging.getLogger(name)
    
    # Eğer daha önce handler eklenmemişse ekle (mükerrer logları önler)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
