def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            'success': True,
            'content': content,
            'file_type': 'text',
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }
