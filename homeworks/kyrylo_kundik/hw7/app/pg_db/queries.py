import logging

from sqlalchemy import desc, func

from app.pg_db.models import Apartment, ApartmentImage, SellerInfo


def add_apartment(kwargs, *, db=None):
    try:
        seller_info = kwargs.pop('seller_info')
        images = kwargs.pop('images')
        result = Apartment(
            **kwargs
        )
        if not db:
            from app.main import db
        db.session.add(result)
        db.session.commit()
        apartment_id = result.id

        for image in images:
            img = ApartmentImage(apartment_id, image)
            db.session.add(img)
        seller_info['apartment_id'] = apartment_id
        seller_info = SellerInfo(**seller_info)
        db.session.add(seller_info)
        db.session.commit()

    except Exception as e:
        logging.error(str(e))


def get_apartments_stats():
    from app.main import db
    agg_stats = db.session.query(
        func.avg(Apartment.price_uah),
        func.stddev_samp(Apartment.price_uah),
        func.avg(Apartment.total_square_meters),
        func.stddev_samp(Apartment.total_square_meters),
        func.count(Apartment.id)
    ).all()[0]

    agg_stats = ["{:.2f}".format(float(x)) for x in agg_stats]

    return {
        'mean_price': agg_stats[0],
        'std_price': agg_stats[1],
        'mean_total_square_meters': agg_stats[2],
        'std_total_square_meters': agg_stats[3],
        'total_apartments_number': int(agg_stats[4])
    }


def get_apartments(limit, offset):
    from app.main import db
    query = db.session.query(Apartment).order_by(desc(Apartment.created_at))
    if limit:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)
    return query.all()
