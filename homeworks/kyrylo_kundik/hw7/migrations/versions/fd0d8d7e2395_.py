"""empty message

Revision ID: fd0d8d7e2395
Revises: 2a2d5752d9ee
Create Date: 2019-08-11 13:32:26.714734

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fd0d8d7e2395'
down_revision = '2a2d5752d9ee'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('apartment', sa.Column('price_uah', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('apartment', 'price_uah')
    # ### end Alembic commands ###
