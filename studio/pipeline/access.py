def invited(email,verified,allowed):
    if verified is not True:return False
    value=str(email or '').strip().lower()
    return bool(value) and value in {str(x).strip().lower() for x in allowed}

