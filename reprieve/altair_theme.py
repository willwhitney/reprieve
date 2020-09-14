import altair as alt


def publication():
    colorscheme = 'set1'
    stroke_color = '333'
    title_size = 24
    label_size = 16
    line_width = 5

    return {
        'config': {
            'view': {
                'height': 500,
                'width': 600,
                'strokeWidth': 0,
                'background': 'white',
            },
            'title': {
                'fontSize': title_size,
            },
            'range': {
                'category': {'scheme': colorscheme},
                'ordinal': {'scheme': colorscheme},
            },
            'axis': {
                'titleFontSize': title_size,
                'labelFontSize': label_size,
                'grid': False,
                'domainWidth': 5,
                'domainColor': stroke_color,
                'tickWidth': 3,
                'tickSize': 9,
                'tickCount': 4,
                'tickColor': stroke_color,
                'tickOffset': 0,
            },
            'legend': {
                'titleFontSize': title_size,
                'labelFontSize': label_size,
                'labelLimit': 0,
                'titleLimit': 0,
                'orient': 'top-right',
                'padding': 10,
                'titlePadding': 10,
                'rowPadding': 5,
                'fillColor': 'white',
                'strokeColor': 'black',
                'cornerRadius': 0,
            },
            'rule': {
                'size': 3,
                'color': '999',
                # 'strokeDash': [4, 4],
            },
            'line': {
                'size': line_width,
                'opacity': 0.4
            },
        }
    }


alt.themes.register('publication', publication)
alt.themes.enable('publication')
